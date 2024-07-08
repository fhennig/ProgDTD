# This file has the steps to do to serve a model.

create_modelstore:
	mkdir model_store

clean_modelstore:
	rm -r model_store
	mkdir model_store

archive_model:
	torch-model-archiver --model-name scale_hyperprior_lightning \
	--version 1.0 \
	--model-file scale_hyperprior_lightning.py \
	--serialized-file weights.zip \
	--handler handler.py \
	--export-path model_store \
	--extra-files "blocks.py,scale_hyperprior.py,ProgDTD.py,dataset.py,load.py,train.py,val.py"

serve_model:
	torchserve --start --model-store model_store --models progdtd=scale_hyperprior_lightning.mar

serve_stop:
	torchserve --stop


docker-build:
	docker build -t torchserve_progdtd .

docker-run:
	docker run -p 8080:8080 -p 8081:8081 torchserve_progdtd

docker-kill:
	docker kill $(shell docker ps -q --filter ancestor=torchserve_progdtd )

# run this when run-image is running. A file example_output.png should be created with the processed example image.
test_example:
	curl http://localhost:8080/predictions/progdtd -T example.png --output example_output.png
