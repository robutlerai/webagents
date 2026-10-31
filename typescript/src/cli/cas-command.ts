/**
 * `webagents cas` / `robutler cas`: CASFS projects from the command line.
 *
 *   cas ls <project>[:<path>]
 *   cas cat <project>:<path>
 *   cas mount <project> <path> <source> [--root <sha>]     source: project:<id> | <id> | pdk:<pkg>[@<version>]
 *   cas mount <project> --pdk <pkg>[@<version>]             lands at pdk/<pkg>@<version>
 *   cas unmount <project> <path>
 *   cas update <project> <path> [--root <sha>]
 *   cas mounts <project>
 *
 * Every command goes through `CasFs` (the same calls an agent makes), under
 * the CLI's login. Reads are verified chunk by chunk and cached on this
 * machine (`~/.cache/robutler/casfs`), so a PDK read once is read once for
 * every project that mounts it.
 */

import { CasError, CasFs, type CasHandle, type MountSourceInput } from '../casfs/index.js';

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function parseTarget(arg: string): { project: string; path: string } {
  const i = arg.indexOf(':');
  const project = (i < 0 ? arg : arg.slice(0, i)).toLowerCase();
  if (!UUID.test(project)) throw new CasError('Cas:PathInvalid', `expected <project id>[:<path>], got ${arg}`);
  return { project, path: i < 0 ? '' : arg.slice(i + 1) };
}

export function parseSource(arg: string): MountSourceInput {
  if (arg.startsWith('pdk:')) {
    const [pkg, version] = arg.slice(4).split('@');
    if (!pkg) throw new CasError('Cas:MountSource', 'pdk:<pkg>[@<version>]');
    return { kind: 'pdk', pkg, ...(version ? { version } : {}) };
  }
  const id = arg.startsWith('project:') ? arg.slice(8) : arg;
  if (!UUID.test(id)) throw new CasError('Cas:MountSource', 'a source is project:<id>, <id> or pdk:<pkg>[@<version>]');
  return { kind: 'project', id: id.toLowerCase() };
}

export interface CasIo {
  out(line: string): void;
  write(bytes: Uint8Array): void;
}

/** Run one `cas` subcommand on an opened CasFs. Returns the JSON document `--json` prints. */
export async function runCas(fs: CasFs, sub: string, args: string[], opts: { root?: string; pdk?: string }, io: CasIo): Promise<unknown> {
  const open = (p: string): Promise<CasHandle> => fs.open(p);
  switch (sub) {
    case 'ls': {
      const t = parseTarget(args[0] ?? '');
      const rows = await (await open(t.project)).list(t.path);
      for (const r of rows) {
        const kind = r.type === 'mount' ? `mount ${r.source?.kind === 'pdk' ? `${r.source.pkg}@${r.source.version}` : r.source?.id} ${r.root?.slice(0, 12)}` : r.type;
        io.out(`${r.type === 'file' ? String(r.size).padStart(12) : ''.padStart(12)}  ${kind.padEnd(6)}  ${r.name}${r.type === 'dir' || r.type === 'mount' ? '/' : ''}`);
      }
      return rows;
    }
    case 'cat': {
      const t = parseTarget(args[0] ?? '');
      if (!t.path) throw new CasError('Cas:PathInvalid', 'cat <project>:<path>');
      const h = await open(t.project);
      for await (const part of h.readStream(t.path)) io.write(part);
      return { path: t.path };
    }
    case 'mounts': {
      const t = parseTarget(args[0] ?? '');
      const m = await (await open(t.project)).mounts();
      for (const x of m) io.out(`${x.path}  ${x.source.kind === 'pdk' ? `${x.source.pkg}@${x.source.version}` : x.source.id}  ${x.root.slice(0, 12)}`);
      return m;
    }
    case 'mount': {
      const t = parseTarget(args[0] ?? '');
      const h = await open(t.project);
      if (opts.pdk) {
        const [pkg, version] = opts.pdk.split('@');
        const r = await h.ensurePdkMount({ pkg, ...(version ? { version } : {}), commit: true });
        io.out(`${r.added ? 'Mounted' : 'Already mounted'} ${r.path}`);
        return r;
      }
      if (args.length < 3) throw new CasError('Cas:PathInvalid', 'mount <project> <path> <source>, or mount <project> --pdk <pkg>[@<version>]');
      const r = await h.mount(args[1], parseSource(args[2]), opts.root ?? null);
      io.out(`Mounted ${r.path} at ${r.entry.root.slice(0, 12)}`);
      return r;
    }
    case 'unmount': {
      const t = parseTarget(args[0] ?? '');
      const r = await (await open(t.project)).unmount(args[1] ?? '');
      io.out(`Unmounted ${r.path}`);
      return r;
    }
    case 'update': {
      const t = parseTarget(args[0] ?? '');
      const r = await (await open(t.project)).updateMount(args[1] ?? '', opts.root ?? null);
      io.out(`${r.path}: ${r.from.slice(0, 12)} -> ${r.entry.root.slice(0, 12)}`);
      return r;
    }
    default:
      throw new CasError('Cas:PathInvalid', `unknown cas command ${sub}: ls, cat, mounts, mount, unmount, update`);
  }
}
