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
# At runtime, listens on :7070 (HTTP) and :9100 (Prometheus). mTLS is enforced
# in cloud via WEBAGENTS_EXECUTOR_MTLS_{KEY,CERT,CA} env vars.

FROM node:20-slim AS base
RUN npm install -g pnpm@9 \
 && apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates python3 \
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
FROM node:20-slim AS runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates dumb-init python3 \
 && rm -rf /var/lib/apt/lists/* \
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
CMD ["node", "dist/cli.js", "--port", "7070", "--metrics-port", "9100"]
