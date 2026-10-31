---
title: Project Files (CASFS)
description: Read and write a project's files by path, verified by hash, with mounts of other projects and PDKs
---

# Project Files (CASFS)

A project on Robutler keeps its files as a tree of content-addressed blobs: every file is named by the SHA-256 of its bytes, every folder by the SHA-256 of its directory manifest, and the project's state is one root hash that moves by compare-and-swap. `CasFs` reads and writes that tree from an agent, a script or the command line, in TypeScript and in Python, with the same operations, names and errors as an app's `host.cas`.

Every read is verified: a folder against its name, a chunk of a large file against its own hash, a small file against its hash. A store that returns the wrong bytes is refused by name and nothing it sent is used or cached. Large files are read in 4 MiB chunks, so a range read moves only the chunks it covers.

## Opening a project

```typescript
import { CasFs } from 'webagents/casfs';
import { DirChunkCache } from 'webagents/casfs/local';

const fs = CasFs.platform({ baseUrl: 'https://robutler.ai', token, cache: new DirChunkCache() });
const p = await fs.open(projectId);
await p.list('cells');
const bytes = await p.read('cells/inv/layout.json', { offset: 0, length: 4096 });
await p.write('runs/r1/report.txt', 'clean');
await p.commit();
```

```python
from webagents.casfs import CasFs, DirChunkCache

fs = CasFs.platform("https://robutler.ai", token, cache=DirChunkCache())
p = await fs.open(project_id)
await p.list("cells")
data = await p.read("cells/inv/layout.json", 0, 4096)
await p.write("runs/r1/report.txt", "clean")
await p.commit()
```

`token` is an agent's API key or your CLI login. To read a project you were given access to for one task, pass the CASFS grant you received: `grants: { [projectId]: grant }` (TypeScript) or `grants={project_id: grant}` (Python). A grant names its project, its path prefixes and its operations, and is accepted only from the account it was issued to.

`commit()` moves the head from the version you opened. If someone else committed meanwhile, the two changes are merged by path: a path only one side changed takes that side; a path both sides changed is refused as `Cas:PathConflict` with the path, and nothing is renamed or dropped.

A project can also be kept in a local directory, in the same layout (`cas/<ab>/<sha256>` and `heads/<projectId>`): `fs.openLocal(id, new LocalDirBackend(dir), new LocalHeads(dir, id))`, or `open_local` in Python.

## Mounts

A mount shows another project, or a published PDK, inside a project as a read-only folder pinned to one version of the source:

```json
"sky130A@2026.10": { "type": "mount", "readOnly": true, "root": "<sha256>",
                     "source": { "kind": "pdk", "id": "<package project id>", "pkg": "sky130A", "version": "2026.10" } }
```

- Reads go through the mount to the source's tree at the pinned root, under **your** access to the source. A mount never gives anyone access to a source they could not read before.
- Nothing inside a mount can be written, created, renamed or deleted (`Cas:MountReadOnly`). The mount itself can be renamed or removed.
- The pin moves only when you ask (`updateMount`). A published PDK version never changes; mount the newer version beside it.
- Mounts nest up to four deep (`Cas:MountDepth`). A chain that leads back to a project already in it is refused (`Cas:MountCycle`).
- The bytes of a mounted source are cached once per machine, so every project that mounts the same PDK version shares them.

| Operation | TypeScript | Python |
| --- | --- | --- |
| list mounts | `p.mounts(root?)` | `p.mounts(root=None)` |
| mount a project | `p.mount(path, { kind: 'project', id }, root?)` | `p.mount(path, {"kind": "project", "id": id}, root=None)` |
| mount a PDK | `p.ensurePdkMount({ pkg, version?, commit: true })` | `p.ensure_pdk_mount(pkg, version=None, commit=True)` |
| unmount | `p.unmount(path)` | `p.unmount(path)` |
| move a pin | `p.updateMount(path, root?)` | `p.update_mount(path, root=None)` |
| place a resolved mount with other changes | `p.placeMount(path, entry)` then `p.commit()` | `p.place_mount(path, entry)` then `p.commit()` |

A PDK always mounts at `pdk/<pkg>@<version>`, one mount per version, shared by everything in the project that uses it. `ensurePdkMount` answers the existing mount when the project already has one, so calling it whenever a cell is given a process is safe. Without `commit`, it answers `{ path, entry }` to place yourself, so the mount lands in the same commit as the change that needs it.

A local directory project stores a mount as the same pointer. Reading through it uses a local package store first (`packagesDir/<sourceId>/cas/...`, for machines with no network), then the platform under your access to the source. A mount that cannot be reached anywhere is `Cas:SourceOffline`, never an empty folder.

Every file read through a mount is listed in `opened` by its full path in the project, and `runRecord(...)` (`run_record`) names every mount a run went through.

## HTTP API

Every request carries `Authorization: Bearer <token>` and, for a grant, `X-Casfs-Grant: <grant>`. Refusals are `{ "ok": false, "error": { "code", "message", "args" } }`.

| Request | Body | Answer |
| --- | --- | --- |
| `GET /api/casfs/projects/<id>` | | `{ ok, head, counter }` |
| `GET /api/casfs/projects/<id>/blobs/<sha>` (`Range` supported) | | the bytes |
| `PUT /api/casfs/projects/<id>/blobs/<sha>` | the bytes | `{ ok, sha256, size }` |
| `POST /api/casfs/projects/<id>/commit` | `{ base, root, message? }` | `{ ok, head, counter }`, or `Cas:HeadMoved` with `args.head` |
| `GET /api/casfs/projects/<id>/versions?limit=` | | `{ ok, versions }` |
| `GET /api/casfs/projects/<id>/mounts?root=` | | `{ ok, root, mounts: [{ path, source, root }] }` |
| `POST /api/casfs/projects/<id>/mounts` | `{ path, source, root? }` | `{ ok, head, path, entry }` |
| `POST /api/casfs/projects/<id>/mounts` | `{ pdk: { pkg, version?, id? }, commit? }` | `{ ok, head, path, entry, added }` |
| `PATCH /api/casfs/projects/<id>/mounts` | `{ path, root? }` | `{ ok, head, path, entry, from }` |
| `DELETE /api/casfs/projects/<id>/mounts?path=` | | `{ ok, head, path }` |

A `source` is `{ "kind": "project", "id": "<project id>" }` or `{ "kind": "pdk", "pkg": "<name>", "version": "<version>", "id": "<package id>" }` (`version` and `id` optional when mounting). A PDK is found by name among packages you own and packages the platform publishes; any other package is mounted by its `id`.

## Errors

| Code | Meaning |
| --- | --- |
| `Cas:NotFound` | no such project, path or blob you can read |
| `Cas:PathInvalid` | a name or path breaks the path rules |
| `Cas:ManifestInvalid` | a manifest is not valid `casfs.dir/1` |
| `Cas:HashMismatch`, `Cas:ChunkMismatch` | bytes did not match their hash; nothing was used |
| `Cas:HeadMoved` | the head moved since the base you named |
| `Cas:PathConflict` | the same path changed on both sides |
| `Cas:Exists` | the path is taken |
| `Cas:LocalOnly` | the file is kept on its owner's machine |
| `Cas:GrantScope`, `Cas:GrantInvalid` | the grant does not allow this |
| `Cas:MountReadOnly` | the path is inside a mount |
| `Cas:MountSource` | the source cannot be read by you, or the version is not published |
| `Cas:MountCycle` | the mount would lead back to a project already in the chain |
| `Cas:MountDepth` | mounts would nest more than four deep |
| `Cas:NotAMount` | the path is not a mount |
| `Cas:SourceOffline` | the source cannot be reached from here |

## Command line

```bash
robutler cas ls <project>[:<path>]
robutler cas cat <project>:<path>
robutler cas mounts <project>
robutler cas mount <project> <path> project:<id> [--root <sha>]
robutler cas mount <project> --pdk sky130A[@<version>]
robutler cas unmount <project> <path>
robutler cas update <project> <path> [--root <sha>]
```

`webagents cas ...` is the same command. With `--json`, each prints one JSON document. Reads are cached in `~/.cache/robutler/casfs` (or `$XDG_CACHE_HOME/robutler/casfs`).
