# ======================================================
# k8s-data-collector Dockerfile
# Builds the Rust telemetry data collector
# ======================================================

############################
# 1) Builder Stage
############################
FROM rust:1.72-bookworm AS builder

# Set working directory
WORKDIR /workspace

# Install build dependencies (system libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy Cargo manifests for dependency caching
COPY Cargo.toml Cargo.lock ./

# Create a dummy main to cache deps
RUN mkdir -p src && echo "fn main() {}" > src/main.rs
RUN cargo build --release || true

# Copy full source
COPY . .

# Build the real collector
RUN cargo build --release

############################
# 2) Runtime Stage
############################
FROM debian:bookworm-slim AS runtime

# Essential runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG COLLECTOR_USER=collector
ARG COLLECTOR_UID=1000
ARG COLLECTOR_GID=1000

RUN groupadd -g ${COLLECTOR_GID} ${COLLECTOR_USER} \
 && useradd -m -u ${COLLECTOR_UID} -g ${COLLECTOR_GID} ${COLLECTOR_USER}

# Copy the compiled binary
WORKDIR /app
COPY --from=builder /workspace/target/release/k8s-data-collector .

# Set ownership
RUN chown ${COLLECTOR_USER}:${COLLECTOR_USER} /app/k8s-data-collector

USER ${COLLECTOR_USER}

ENTRYPOINT ["/app/k8s-data-collector"]
CMD ["--help"]
