import assert from 'node:assert/strict'

import { test } from 'vitest'

import { rebasePath, resolveRenamePath } from './git-review-ops'

test('resolveRenamePath: plain path is unchanged', () => {
  assert.equal(resolveRenamePath('src/a.ts'), 'src/a.ts')
})

test('resolveRenamePath: simple rename resolves to the new path', () => {
  assert.equal(resolveRenamePath('old.ts => new.ts'), 'new.ts')
})

test('resolveRenamePath: brace rename resolves to the new path', () => {
  assert.equal(resolveRenamePath('src/{old => new}/file.ts'), 'src/new/file.ts')
})

test('resolveRenamePath: brace rename collapsing a segment', () => {
  assert.equal(resolveRenamePath('src/{lib => }/file.ts'), 'src/file.ts')
})

// rebasePath maps repo-root-relative paths (what simple-git returns) to
// cwd-relative paths (what the renderer expects). When the session cwd is a
// subdirectory of the repo root, un-rebased paths would cause wrong-file
// diffs and stage/unstage failures.

test('rebasePath: nested cwd strips the common prefix', () => {
  assert.equal(rebasePath('/repo', '/repo/apps', 'apps/foo.ts'), 'foo.ts')
})

test('rebasePath: deeply nested cwd strips all common segments', () => {
  assert.equal(rebasePath('/repo', '/repo/apps/desktop', 'apps/desktop/src/a.ts'), 'src/a.ts')
})

test('rebasePath: cwd === root is a no-op (no subdirectory)', () => {
  assert.equal(rebasePath('/repo', '/repo', 'src/a.ts'), 'src/a.ts')
})

test('rebasePath: falsy inputs pass through unchanged', () => {
  assert.equal(rebasePath(null, '/repo', 'src/a.ts'), 'src/a.ts')
  assert.equal(rebasePath('/repo', null, 'src/a.ts'), 'src/a.ts')
  assert.equal(rebasePath('/repo', '/repo', ''), '')
})
