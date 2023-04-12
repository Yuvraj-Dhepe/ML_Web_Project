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
    - Build an image via use of DockerFile
    - Run the image in a container by docker run -p 8080:80 <image-name>
    - Here for example, if you have an existing container named my-app that is listening on port 80, and you want to expose it on port 8080.
    - This will link the container to port 8080 on the host machine, so you can access it by visiting http://localhost:8080 in your web browser.
- Setup Github workflow
- Setup AWS IAM role
    - Create a new user and attach the below permissions to the user.
    - Provide the AmazonEC2ContainerRegistryFullAccess & AmazonEC2FullAccess permissions
    - Setup access keys for the user
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
        


    