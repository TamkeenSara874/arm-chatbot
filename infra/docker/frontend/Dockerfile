# Production image for the React/Vite frontend. Multi-stage: build the static
# bundle in a node image, serve it with nginx (no node runtime in the final image).
FROM node:20-alpine AS deps
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

FROM deps AS build
COPY frontend/ .
# Vite inlines VITE_* env vars into the bundle AT BUILD TIME -- these must be
# passed as --build-arg, not runtime env vars, or the deployed bundle will
# still point at whatever was baked in here (see docs/runbook.md).
ARG VITE_API_URL=""
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

FROM nginx:1.27-alpine AS production
COPY infra/docker/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -q -O /dev/null http://localhost:8080/ || exit 1
CMD ["nginx", "-g", "daemon off;"]
