![phData Logo](phData.png "phData Logo")

# Sound Realty Housing Pricing

To create the Kubernetes cluster on your local machine, run:

```sh
 minikube start
```

To build Docker images directly inside the Minikube cluster (without needing a Docker registry), configure your shell to use Minikube’s Docker daemon:
```sh
eval $(minikube docker-env)
```

Then, to build the image with the `1.0.0` tag, run:

```sh
docker build -t housing-model:1.0.0 .
```

Alternatively, you can use a CI/CD pipeline to automate the build and deployment process. Please, check out [deploy.yml](https://github.com/icleveston/sound-realty/blob/master/.github/workflows/deploy.yml).


Open a new terminal and mount the local directory containing the data and model weights into the cluster:

```sh
 minikube mount /home/brain/PycharmProjects/phData/data:/data
```

Apply the Kubernetes configurations to create the volumes, pods, services, and horizontal autoscaler:
```sh
 kubectl apply -f k8s/volume.yaml
 kubectl apply -f k8s/deployment.yaml
 kubectl apply -f k8s/service.yaml
 kubectl apply -f k8s/hpa.yaml
```


To access the service within the cluster, run:

```sh
 minikube service housing-model-service
```

This will open the service in your default browser. You can access the API documentation at:

```sh
 http://192.168.49.2:32622/docs
```