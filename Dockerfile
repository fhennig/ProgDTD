# Use an official PyTorch image with CUDA support as a parent image
FROM pytorch/torchserve:0.11.0-cpu

USER root
# Install libjpeg and libpng
RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    libpng-dev
USER model-server

# Install any needed packages specified in req.txt
COPY req.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U torchtext


# Copy the TorchServe config file
COPY config.properties /home/model-server/config.properties

# Copy the .mar file to the model store
COPY model_store/scale_hyperprior_lightning.mar /home/model-server/model-store/

# Expose ports for TorchServe
EXPOSE 8080
EXPOSE 8081

# Define environment variables
ENV MODEL_STORE=/home/model-server/model-store
ENV TS_CONFIG_FILE=/home/model-server/config.properties
ENV MANAGEMENT_API=true

# Start TorchServe
CMD ["torchserve", "--start", "--model-store", "model-store", "--models", "progdtd=scale_hyperprior_lightning.mar"]
