ARG SPARK_VERSION=3.1.3
ARG STITCHING_SPARK_GIT_TAG=1.10.0

# Temporary container for building cleanly
FROM janeliascicomp/builder:1.2.1 as builder
ARG STITCHING_SPARK_GIT_TAG

# Checkout and build the code
WORKDIR /tmp/app
RUN git clone --branch ${STITCHING_SPARK_GIT_TAG} --depth 1 https://github.com/saalfeldlab/stitching-spark.git . \
    && git submodule update --init --recursive \
    && /usr/local/bin/buildinfo.sh \
    && mvn package

# Find the built jar, based on the version in the pom file
RUN xq -r '.project.artifactId+"-"+.project.version+".jar"' pom.xml > filename \
    && mv /tmp/app/target/`cat filename` app.jar

RUN echo "$STITCHING_SPARK_GIT_TAG" > /tmp/app/VERSION

# Create final image
FROM ghcr.io/janeliascicomp/spark:${SPARK_VERSION}
ARG STITCHING_SPARK_GIT_TAG

LABEL \
    org.opencontainers.image.title="Stitching Spark" \
    org.opencontainers.image.description="Reconstructing large microscopy images from overlapping image tiles on a high-performance Spark cluster" \
    org.opencontainers.image.authors="rokickik@janelia.hhmi.org" \
    org.opencontainers.image.licenses="BSD-3-Clause AND GPL-2.0" \
    org.opencontainers.image.version=${STITCHING_SPARK_GIT_TAG}

WORKDIR /app
COPY --from=builder /tmp/app/LICENSE.txt /app/LICENSE
COPY --from=builder /tmp/app/VERSION /app
COPY --from=builder /tmp/app/app.jar /app
COPY --from=builder /buildinfo /app
