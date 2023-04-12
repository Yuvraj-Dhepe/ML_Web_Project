## Notes

### T1: Github and Code Setup
- Setup the git repo.
    - New Environment
    - Setup.py: Building our application as a package.
    - Requirements.txt
    - Creation of src for building of the ML as a package. [25.10] entire project development happens in src folder and it's building, likewise we have every folder that will act as a package.

- Created the src folder and built the package, by using requirements.txt

### T2: Project Structure, Logging and Exception Handling
- Creating the entire project structure, do logging, doing exception handling.  

### T3: Project Problem Statement, EDA and Model Training
- Use jupyter notebook to do EDA and observe.
- Use these observations to be put into the .py files and providing to the stakeholder
- There has to be a reason for every bit that you do in the project.
- Feature engineering is nothing but the process of creating new features from the existing features.
- Feature selection is nothing but the process of selecting the best features from the existing features.


### T9: Project deployment in AWS cloud using CICD pipelines
- Having Elastic Beanstalk, is a kind of an instance that will be provided where one can deploy entire application.
- 2 very important configurations that needs to be setup when we are working with elastic beanstalk are .ebextensions and setting up application.py

### T11: Deploying the project using AWS
- Setup Docker container:
    - Build an image using Dockerfile
    - Run the image in a container using the command docker run -p 8088:80 -v /path/to/ml_application:/app my-app
        - The -p flag is used to expose a port on the host machine, so that you can access it from outside the container. Here, we are exposing port 8088 on the host machine, which is mapped to port 80 on the container.
        - The -v flag is used to mount a volume, which allows the container to access a directory on the host machine. The first path is the path to the directory on the host machine, and the second path is the path to the directory in the container.
        - The last argument my-app is the name of the image that you want to run.
        - The container is listening on port 80, so the application.py file should have the port number set to 80.
        - Access the application by visiting http://localhost:8088 in your web browser.
        - If you want to run the container in the background, you can use the -d flag.
            
    - Here for example, if you have an existing container named my-app that is listening on port 80, and you want to expose it on port 8080.
    - This will link the container to port 8080 on the host machine, so you can access it by visiting http://localhost:8080 in your web browser.
- Setup Github workflow
- Setup AWS IAM role
    - Create a new user and attach the below permissions to the user.
    - Provide the AmazonEC2ContainerRegistryFullAccess & AmazonEC2FullAccess permissions
    - Setup access keys for the user and also download it in csv format.

- Go to Elastic Container Registry and create a new repository named student performance and copy the URL for the repository to the aws yml file.
- Go to EC2 and setup a new instance.
    - Run the instance with default settings, just set all the HTTP connections.
    - Once the instance is up and running, go to the instance and copy the public IP address.
    - Go to the terminal of instance to install docker.
        - sudo apt-get update -y
        - sudo apt-get upgrade
        - curl -fsSL https://get.docker.com -o get-docker.sh
        - sudo sh get-docker.sh
        - sudo usermod -aG docker ubuntu
        - newgrp docker

        Our main aim is that once we update our code, from github, our docker image should go to the ecr repository, and then this docker image will get installed in the ec2 instance, that we have created.
- Now setup runner in github, to run the workflow. This runner will trigger the workflow, whenever there is a change in the code.
    - Simply create a self-hosted runner in github runner
    - Go to the instance and run all the commands to setup the runner, as per github, the name is given as self-hosted. 
    - For the default options where it asks for anything simply leave it blank.

- After this add the github secrets in actions with the following keys:
    - AWS_ACCESS_KEY_ID: Created when we created the user in IAM
    - AWS_SECRET_ACCESS_KEY: Created when we created the user in IAM
    - AWS_DEFAULT_REGION: See it in the ec2 instance details
    - AWS_ECR_LOGIN_URI: See it in the ecr repository details #  kind of format. but it doesn't include the repository name.
    - AWS_ECR_REPOSITORY: studentperformance or whatever name you have given to the repository