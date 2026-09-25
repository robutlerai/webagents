/**
 * Syntax colour for the code blocks an agent writes (2026-09-24).
 *
 * Deliberately small: comments, strings, numbers, keywords, function names and
 * types are what make code readable at a glance, and a tokenizer per language
 * family gets those right for the languages agents write most. Anything it does
 * not know is shown plain rather than guessed at. Block comments and multi-line
 * strings carry across lines through `HighlightState`, because a code block is
 * highlighted line by line.
 */

import type { Theme } from './theme';

type Family = 'c' | 'py' | 'sh' | 'json' | 'yaml' | 'diff' | 'sql' | 'html' | 'css' | 'plain';

const ALIASES: Record<string, Family> = {
  js: 'c', javascript: 'c', jsx: 'c', mjs: 'c', cjs: 'c', ts: 'c', typescript: 'c', tsx: 'c',
  java: 'c', c: 'c', h: 'c', cpp: 'c', 'c++': 'c', cc: 'c', hpp: 'c', cs: 'c', csharp: 'c',
  go: 'c', golang: 'c', rust: 'c', rs: 'c', swift: 'c', kotlin: 'c', kt: 'c', scala: 'c', dart: 'c', php: 'c',
  py: 'py', python: 'py', python3: 'py', rb: 'py', ruby: 'py',
  sh: 'sh', bash: 'sh', zsh: 'sh', shell: 'sh', console: 'sh', fish: 'sh', powershell: 'sh', ps1: 'sh', dockerfile: 'sh',
  json: 'json', jsonc: 'json', json5: 'json',
  yaml: 'yaml', yml: 'yaml', toml: 'yaml', ini: 'yaml', env: 'yaml',
  diff: 'diff', patch: 'diff',
  sql: 'sql', postgres: 'sql', postgresql: 'sql', mysql: 'sql', sqlite: 'sql',
  html: 'html', xml: 'html', svg: 'html', vue: 'html',
  css: 'css', scss: 'css', less: 'css',
};

const C_KEYWORDS = new Set(
  (
    'abstract as async await break case catch class const continue debugger default defer delete do else enum ' +
    'export extends false final finally fn for from func function go if impl implements import in instanceof interface ' +
    'let loop match mod module mut namespace new nil null package private protected pub public readonly return ' +
    'select self static struct super switch this throw throws trait true try type typeof undefined union unsafe use ' +
    'using var void where while with yield'
  ).split(' '),
);

const PY_KEYWORDS = new Set(
  (
    'and as assert async await break class continue def del elif else except False finally for from global if ' +
    'import in is lambda None nonlocal not or pass raise return self True try while with yield match case ' +
    'begin end module require then unless until do puts'
  ).split(' '),
);

const SH_KEYWORDS = new Set(
  'if then else elif fi for in do done while until case esac function return export local readonly declare set unset source alias exit'.split(
    ' ',
  ),
);

const SQL_KEYWORDS = new Set(
  (
    'select from where and or not insert into values update set delete create table index view drop alter add ' +
    'column primary key foreign references join left right inner outer full on group by order having limit offset ' +
    'as distinct union all case when then else end null is in exists between like returning with begin commit rollback'
  ).split(' '),
);

export interface HighlightState {
  /** Inside a block comment, a multi-line string, or a template literal. */
  open?: { kind: 'comment' | 'string'; close: string };
}

export function familyOf(lang: string): Family {
  return ALIASES[lang.trim().toLowerCase()] ?? 'plain';
}

interface Token {
  text: string;
  kind?: keyof Theme['palette']['code'];
}

/** One line of code, coloured. `state` is read and updated for the next line. */
export function highlightLine(theme: Theme, lang: string, line: string, state: HighlightState): string {
  const { paint, palette } = theme;
  if (!paint.on) return line;
  const tokens = tokenize(familyOf(lang), line, state);
  return tokens
    .map((token) => {
      if (!token.kind) return paint.fg(palette.text, token.text);
      const colour = palette.code[token.kind];
      return token.kind === 'comment' ? paint.italic(paint.fg(colour, token.text)) : paint.fg(colour, token.text);
    })
    .join('');
}

function tokenize(family: Family, line: string, state: HighlightState): Token[] {
  if (family === 'plain') return [{ text: line }];
  if (family === 'diff') return [diffToken(line)];
  const out: Token[] = [];
  let i = 0;
  const push = (text: string, kind?: Token['kind']) => {
    if (text) out.push({ text, kind });
  };

  // Continue whatever the previous line left open.
  if (state.open) {
    const end = line.indexOf(state.open.close);
    if (end === -1) {
      push(line, state.open.kind);
      return out;
    }
    push(line.slice(0, end + state.open.close.length), state.open.kind);
    i = end + state.open.close.length;
    state.open = undefined;
  }

  const commentStart = family === 'c' || family === 'css' ? '//' : family === 'sql' ? '--' : family === 'html' ? null : '#';
  let atLineStart = line.slice(0, i).trim() === '';
  while (i < line.length) {
    const rest = line.slice(i);
    const ch = line[i];

    // Comments.
    if (family === 'html' && rest.startsWith('<!--')) {
      const end = rest.indexOf('-->');
      if (end === -1) {
        push(rest, 'comment');
        state.open = { kind: 'comment', close: '-->' };
        return out;
      }
      push(rest.slice(0, end + 3), 'comment');
      i += end + 3;
      continue;
    }
    if ((family === 'c' || family === 'css') && rest.startsWith('/*')) {
      const end = rest.indexOf('*/', 2);
      if (end === -1) {
        push(rest, 'comment');
        state.open = { kind: 'comment', close: '*/' };
        return out;
      }
      push(rest.slice(0, end + 2), 'comment');
      i += end + 2;
      continue;
    }
    if (commentStart && rest.startsWith(commentStart) && !(family === 'css' && commentStart === '//')) {
      // `#` starts a shell comment only at a word boundary (not `a#b`, not `$#`).
      if (commentStart !== '#' || i === 0 || /\s/.test(line[i - 1])) {
        push(rest, 'comment');
        return out;
      }
    }

    // Strings.
    if (family === 'py' && (rest.startsWith('"""') || rest.startsWith("'''"))) {
      const quote = rest.slice(0, 3);
      const end = rest.indexOf(quote, 3);
      if (end === -1) {
        push(rest, 'string');
        state.open = { kind: 'string', close: quote };
        return out;
      }
      push(rest.slice(0, end + 3), 'string');
      i += end + 3;
      continue;
    }
    if (ch === '"' || ch === "'" || (ch === '`' && family === 'c')) {
      let j = i + 1;
      while (j < line.length && line[j] !== ch) j += line[j] === '\\' ? 2 : 1;
      if (j >= line.length && ch === '`') {
        push(rest, 'string');
        state.open = { kind: 'string', close: '`' };
        return out;
      }
      const literal = line.slice(i, Math.min(line.length, j + 1));
      // A JSON or YAML string followed by a colon is a key.
      const isKey = (family === 'json' || family === 'yaml') && /^\s*:/.test(line.slice(j + 1));
      push(literal, isKey ? 'property' : 'string');
      i += literal.length;
      atLineStart = false;
      continue;
    }

    // Numbers.
    const number = /^(0x[0-9a-fA-F_]+|\d[\d_]*(\.\d+)?([eE][+-]?\d+)?[a-zA-Z%]*)/.exec(rest);
    if (number && (i === 0 || !/[\w$]/.test(line[i - 1]))) {
      push(number[0], 'number');
      i += number[0].length;
      atLineStart = false;
      continue;
    }

    // Shell variables and flags.
    if (family === 'sh') {
      const variable = /^\$(\{[^}]*\}|[A-Za-z_]\w*|[@#?$!*0-9])/.exec(rest);
      if (variable) {
        push(variable[0], 'property');
        i += variable[0].length;
        continue;
      }
      const flag = /^--?[A-Za-z][\w-]*/.exec(rest);
      if (flag && (i === 0 || /\s/.test(line[i - 1]))) {
        push(flag[0], 'type');
        i += flag[0].length;
        continue;
      }
    }

    // HTML tags and attributes.
    if (family === 'html') {
      const tag = /^<\/?[A-Za-z][\w:-]*|^\/?>/.exec(rest);
      if (tag) {
        push(tag[0], 'keyword');
        i += tag[0].length;
        continue;
      }
      const attr = /^[A-Za-z_:][\w:.-]*(?==)/.exec(rest);
      if (attr) {
        push(attr[0], 'property');
        i += attr[0].length;
        continue;
      }
    }

    // CSS properties and colours.
    if (family === 'css') {
      const property = /^[a-z-]+(?=\s*:)/.exec(rest);
      if (property) {
        push(property[0], 'property');
        i += property[0].length;
        continue;
      }
      const hex = /^#[0-9a-fA-F]{3,8}\b/.exec(rest);
      if (hex) {
        push(hex[0], 'number');
        i += hex[0].length;
        continue;
      }
    }

    // Decorators and annotations.
    if (ch === '@' && (family === 'py' || family === 'c')) {
      const decorator = /^@[\w.]+/.exec(rest);
      if (decorator) {
        push(decorator[0], 'func');
        i += decorator[0].length;
        continue;
      }
    }

    // Words.
    const word = /^[A-Za-z_$][\w$]*/.exec(rest);
    if (word) {
      const w = word[0];
      const after = line.slice(i + w.length);
      let kind: Token['kind'];
      if (family === 'yaml' && atLineStart && /^\s*[:=]/.test(after)) kind = 'property';
      else if (isKeyword(family, w)) kind = 'keyword';
      else if (family === 'sh' && atLineStart) kind = 'func';
      else if (/^\s*\(/.test(after)) kind = 'func';
      else if (/^[A-Z][a-z]\w*$/.test(w) && family !== 'sql') kind = 'type';
      push(w, kind);
      i += w.length;
      atLineStart = false;
      continue;
    }

    if (!/\s/.test(ch)) atLineStart = family === 'sh' && /[|;&]/.test(ch);
    push(ch, /[{}()[\];,.:]/.test(ch) ? 'punctuation' : undefined);
    i += 1;
  }
  return out;
}

function isKeyword(family: Family, word: string): boolean {
  switch (family) {
    case 'c':
      return C_KEYWORDS.has(word);
    case 'py':
      return PY_KEYWORDS.has(word);
    case 'sh':
      return SH_KEYWORDS.has(word);
    case 'sql':
      return SQL_KEYWORDS.has(word.toLowerCase());
    case 'json':
    case 'yaml':
      return word === 'true' || word === 'false' || word === 'null' || word === 'yes' || word === 'no';
    default:
      return false;
  }
}

function diffToken(line: string): Token {
  if (line.startsWith('+++') || line.startsWith('---') || line.startsWith('diff ') || line.startsWith('index ')) {
    return { text: line, kind: 'punctuation' };
  }
  if (line.startsWith('+')) return { text: line, kind: 'added' };
  if (line.startsWith('-')) return { text: line, kind: 'removed' };
  if (line.startsWith('@@')) return { text: line, kind: 'type' };
  return { text: line };
}
