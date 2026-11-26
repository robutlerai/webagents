#!/usr/bin/env node
// The `robutler` packages (packages/robutler-npm, packages/robutler-py) hold no code: each depends on exactly one
// version of webagents, its own. This script checks that, and can write it.
//
//   node scripts/thin-packages.mjs                    check both against typescript/package.json and python/pyproject.toml
//   node scripts/thin-packages.mjs --write            set both thin packages to those versions
//   node scripts/thin-packages.mjs --only npm --npm 1.2.3     check the npm one against a version given here
//   node scripts/thin-packages.mjs --write --pypi 1.2.3       set the PyPI one to a version given here
//   node scripts/thin-packages.mjs --root <dir>       run on another checkout
//
// Exit 0 when everything agrees, 1 with one line per disagreement.
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const args = process.argv.slice(2);
const flag = (name) => args.includes(name);
const value = (name) => (args.includes(name) ? args[args.indexOf(name) + 1] : undefined);
const root = resolve(value('--root') ?? join(dirname(fileURLToPath(import.meta.url)), '..'));
const only = value('--only');
const write = flag('--write');
const SEMVER = /^\d+\.\d+\.\d+(?:[-+.][0-9A-Za-z.-]+)?$/;
const problems = [];
const fail = (line) => problems.push(line);

const pyVersion = (text) => (text.match(/^version = "([^"]+)"$/m) ?? [])[1];

function npm() {
  const main = JSON.parse(readFileSync(join(root, 'typescript/package.json'), 'utf8'));
  const file = join(root, 'packages/robutler-npm/package.json');
  const thin = JSON.parse(readFileSync(file, 'utf8'));
  const want = value('--npm') ?? main.version;
  if (!SEMVER.test(String(want))) return fail(`npm: ${want} is not a version`);
  if (write) {
    thin.version = want;
    thin.dependencies = { webagents: want };
    writeFileSync(file, `${JSON.stringify(thin, null, 2)}\n`);
  }
  if (thin.name !== 'robutler') fail(`npm: packages/robutler-npm is named ${thin.name}, not robutler`);
  if (thin.version !== want) fail(`npm: robutler is ${thin.version}, webagents is ${want}`);
  const deps = Object.entries(thin.dependencies ?? {});
  if (deps.length !== 1 || deps[0][0] !== 'webagents' || deps[0][1] !== want) {
    fail(`npm: robutler must depend on exactly webagents ${want}, and it depends on ${JSON.stringify(thin.dependencies ?? {})}`);
  }
  if (JSON.stringify(thin.bin) !== JSON.stringify({ robutler: 'bin/robutler.js' })) fail('npm: robutler must install one command, robutler, from bin/robutler.js');
  const bin = join(root, 'packages/robutler-npm/bin/robutler.js');
  if (!existsSync(bin) || !/^import 'webagents\/cli\/robutler';$/m.test(readFileSync(bin, 'utf8'))) fail("npm: bin/robutler.js must import 'webagents/cli/robutler'");
  if (main.exports?.['./cli/robutler']?.import !== './dist/cli/robutler.js') fail('npm: webagents must export ./cli/robutler as ./dist/cli/robutler.js');
  if (main.bin?.robutler !== 'dist/cli/robutler.js') fail('npm: webagents no longer ships dist/cli/robutler.js as its robutler command');
}

function pypi() {
  const main = readFileSync(join(root, 'python/pyproject.toml'), 'utf8');
  const file = join(root, 'packages/robutler-py/pyproject.toml');
  let thin = readFileSync(file, 'utf8');
  const want = value('--pypi') ?? pyVersion(main);
  if (!SEMVER.test(String(want))) return fail(`pypi: ${want} is not a version`);
  if (write) {
    thin = thin.replace(/^version = "[^"]*"$/m, `version = "${want}"`).replace(/^dependencies = \[[^\]]*\]$/m, `dependencies = ["webagents==${want}"]`);
    writeFileSync(file, thin);
  }
  if (!/^name = "robutler"$/m.test(thin)) fail('pypi: packages/robutler-py is not named robutler');
  if (pyVersion(thin) !== want) fail(`pypi: robutler is ${pyVersion(thin)}, webagents is ${want}`);
  const deps = (thin.match(/^dependencies = \[([^\]]*)\]$/m) ?? [])[1];
  if (deps === undefined || deps.trim() !== `"webagents==${want}"`) fail(`pypi: robutler must depend on exactly webagents==${want}, and it depends on [${deps ?? ''}]`);
  if (/^\[project\.(scripts|gui-scripts|entry-points)/m.test(thin)) fail('pypi: robutler must install no command of its own; webagents installs robutler');
  if (!/^robutler = "webagents\.robutler_entry:main"$/m.test(main)) fail('pypi: webagents no longer installs the robutler command');
}

if (only !== undefined && only !== 'npm' && only !== 'pypi') fail(`--only takes npm or pypi, not ${only}`);
if (only === undefined || only === 'npm') npm();
if (only === undefined || only === 'pypi') pypi();

if (problems.length) {
  for (const line of problems) console.error(line);
  process.exit(1);
}
console.log(`robutler packages agree with webagents${write ? ' (written)' : ''}`);
