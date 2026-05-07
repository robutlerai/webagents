# syntax=docker/dockerfile:1.6
#
# function-executor image — sandboxed user-code runtime.
#
# Two stages: SDK build (webagents/typescript) + executor image.
# Built from the repo root so we can vendor the SDK without publishing it.
#
# Build with:
#   docker buildx build -f webagents/packages/webagents-executor/Dockerfile \
#     -t function-executor:dev .
#
# At runtime, listens on :7070 (HTTP) and :9100 (Prometheus). Authentication
# on the portal→executor leg is NetworkPolicy ingress (only portal +
# webagentsd may reach :7070); the legacy mTLS shared cert was retired in
# ADR-0007. The reverse leg (executor→portal /api/internal/fn-host) uses
# RS256 JWT bearer auth via lib/agents/fn-host-token.ts.
#
# Node 20 LTS — pinned because `isolated-vm@5.0.4` (latest) only publishes
# prebuilt N-API binaries and source-builds against V8 ABIs through Node
# 22; Node 24 bumped V8 to a C++20-only surface that `isolated-vm` does
# not yet support. Node 20 has linux/arm64 prebuilts, so `pnpm install`
# never invokes node-gyp and image builds finish in seconds. Webcrypto
# `subtle`, `structuredClone`, and `URL`/`URLSearchParams` are all GA in
# Node 20; native `URLPattern` is missing → polyfilled via the
# `urlpattern-polyfill` npm package on the host side and bridged into
# the isolate (see runtimes/js-v1.ts).

FROM node:20-bookworm-slim AS base
# `python3` is the only build-time dep — kept so `node-gyp` works as a
# safety net if `isolated-vm`'s prebuilt is ever missing for the target
# arch. With Node 20 prebuilts available for linux/{amd64,arm64}, the
# install path is a binary copy and never invokes the C++ toolchain in
# practice. `ca-certificates` is needed for any HTTPS fetch (npm
# registry, gyp headers fallback).
RUN npm install -g pnpm@9 \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      python3 \
 && rm -rf /var/lib/apt/lists/*

# --- Stage 1: build the webagents SDK ---------------------------------------
FROM base AS sdk-builder
WORKDIR /sdk
COPY webagents/typescript/package.json webagents/typescript/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile --ignore-scripts
COPY webagents/typescript/tsconfig.json ./
COPY webagents/typescript/src ./src
COPY webagents/typescript/scripts ./scripts
RUN pnpm build

# --- Stage 2: build the executor package -----------------------------------
FROM base AS exec-builder
WORKDIR /exec
COPY webagents/packages/webagents-executor/package.json ./
COPY webagents/packages/webagents-executor/tsconfig.json ./
COPY webagents/packages/webagents-executor/src ./src
COPY webagents/packages/webagents-executor/README.md ./
# Vendor the SDK so the executor's `import 'webagents/executor'` resolves at
# build time without requiring a published npm package.
COPY --from=sdk-builder /sdk/package.json node_modules/webagents/package.json
COPY --from=sdk-builder /sdk/dist node_modules/webagents/dist
RUN pnpm install --ignore-workspace --no-frozen-lockfile --prod=false \
 && pnpm build

# --- Stage 3: minimal runtime image ----------------------------------------
FROM node:20-bookworm-slim AS runtime
# `node:20-bookworm-slim` already ships a `node` user/group at UID/GID 1000, which
# collides with the explicit `groupadd --gid 1000 executor` below. We drop
# the stock user first (no other reference to it in this image) and then
# recreate at the same UID so the cloud Deployment's `runAsUser: 1000`
# keeps working unchanged. Use `|| true` so the build is robust to base-
# image changes that might rename or remove that user upstream.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates dumb-init python3 \
 && rm -rf /var/lib/apt/lists/* \
 && (userdel -r node 2>/dev/null || true) \
 && (groupdel node 2>/dev/null || true) \
 && groupadd --gid 1000 executor \
 && useradd --uid 1000 --gid 1000 --shell /usr/sbin/nologin --create-home executor

WORKDIR /app
ENV NODE_ENV=production \
    EXECUTOR_HTTP_PORT=7070 \
    EXECUTOR_METRICS_PORT=9100

COPY --from=exec-builder /exec/dist ./dist
COPY --from=exec-builder /exec/node_modules ./node_modules
COPY --from=exec-builder /exec/package.json ./

USER executor
EXPOSE 7070 9100

ENTRYPOINT ["dumb-init", "--"]
# `--no-node-snapshot` is required by isolated-vm — the startup snapshot
# is incompatible with the embedder's heap layout and can crash V8 the
# first time we instantiate an Isolate.
CMD ["node", "--no-node-snapshot", "dist/cli.js", "--port", "7070", "--metrics-port", "9100"]
