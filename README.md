# containers

This is a place to put Dockerfiles for bioimage analysis tools, which can be used in bioimage analysis workflows. It is structured in the style of [BioContainers/containers](https://github.com/BioContainers/containers). 

These containers are currently used in [Nextflow Modules](https://github.com/BioImageTools/nextflow-modules), but could also be used in Galaxy and other workflow tools. 

# Principles

We want containers that:

1) Are reproducible - if you rebuild the same Dockerfile a year later, you should get basically the same container. 
2) Are optimized - We use standard techniques to speed up builds and we use multistage builds to reduce the size of the final image. 
3) Include metadata -  every image includes [OCI metadata](https://specs.opencontainers.org/image-spec/annotations/)

# Publishing images

Containers in this repository are published as packages in the  GitHub Container Registry using these steps: 

1. Obtain a GitHub [personal authentication token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) and save it into your terminal environment as `$GITHUB_PACKAGE_TOKEN`. 
   
2. Log into the registry

```bash
echo $GITHUB_PACKAGE_TOKEN | docker login ghcr.io --username $USER --password-stdin
```

3. Build your image

```bash
./build.sh stitching-spark/1.10.0
```

The build script adds important metadata to the image, and then asks you if you want to push it to GCR. 

4. Reference your image

The image will be available in GHCR, but needs to be made public manually. After that, it can be referenced like this:

```bash
ghcr.io/bioimagetools/toolname:1.0.0
```
