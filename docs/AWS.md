AWS Services
================
- `AWS S3`: Amazon S3 or Amazon Simple Storage Service is a service offered by Amazon Web Services that provides object storage through a web service interface.
- `AWS Lambda`: AWS Lambda is an event-driven, serverless computing platform provided by Amazon as a part of Amazon Web Services.
- `AWS Step Functions`: AWS Step Functions is a serverless orchestration service that lets you integrate with AWS Lambda functions and other AWS services to build business-critical applications. Through Step Functions' graphical console, you see your application’s workflow as a series of event-driven steps.
- `AWS SQS`: Amazon Simple Queue Service (Amazon SQS) lets you send, store, and receive messages between software components at any volume, without losing messages or requiring other services to be available.
- `AWS ECR`: Amazon Elastic Container Registry (ECR) is a fully managed Docker container registry that makes it easy to store, share, and deploy container images.
- `AWS API Gateway`: Amazon API Gateway is an AWS service for creating, publishing, maintaining, monitoring, and securing REST, HTTP, and WebSocket APIs at any scale.
- `AWS EC2`: Amazon Elastic Compute Cloud (EC2) is the Amazon Web Service you use to create and run virtual machines in the cloud (we call these virtual machines 'instances')
- `AWS RDS`: Amazon Relational Database Service (Amazon RDS) is a web service that makes it easier to set up, operate, and scale a relational database in the AWS Cloud.
- `AWS DynamoDB`: Amazon DynamoDB is a fully managed NoSQL database service that provides fast and predictable performance with seamless scalability. It is a serverless NoSQL database service that supports key-value and document data models. 
- `AWS CodeCommit`: AWS CodeCommit is a version control service hosted by Amazon Web Services that you can use to privately store and manage assets (such as documents, source code, and binary files) in the cloud.
- `AWS SageMaker`: Amazon SageMaker is a cloud-based machine-learning platform that allows users to construct, design, train, tune, and deploy machine-learning models in a production-ready hosted environment. 
- `AWS Textract`: Amazon Textract is a machine learning (ML) service that automatically extracts text, handwriting, and data from scanned documents. 
- `AWS S3 Glacier`: Amazon S3 Glacier is a low-cost, cloud-archive storage service that provides secure and durable storage for data archiving and online backup.


## AWS CI/CD Deployment

Production Grade CICD Deployment on AWS:

- CI/CD -> Continuous Integration and Continuous Delivery/Continuous Deployment
- Why CI/CD Needed?
    - Simple deployment with ‘AWS Elastic Beanstalk’ and Docker. AWS Elastic Beanstalk automatically allocate memory and it has auto scale functionality. It is having more cost than CI/CD deployment.
    - CI/CD Deployment: We create an application in local system and then commit to GitHub (as an example) and then deployed to AWS CodePipeline (with customization on scaling). When we adding new features or changing some features and commit to GitHub, then the AWS CodePipeline automatically get all the changes/updates from GitHub where we do not need to deploy further.
- Why Docker is needed?
    - Docker helps to install and setup everything of a project in EC2 system/any cloud system automatically and it can be done for n number of system without any manual setup.
- ECR
    - It is similar to DockerHub where we can store image (say Docker image)
- Launch EC2 Instance (machine)
    - Pull Docker image from ECR to EC2
    - Launch Docker image in EC2
- Write a YML file to automate above step through GitHub Actions
    - .github/workflow/cicd.yml
<pre>
name: workflow

on:
  push:
    branches:
      - main
    paths-ignore:
      - 'README.md'

permissions:
  id-token: write
  contents: read

jobs:
  integration:
    name: Continuous Integration
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v3

      - name: Lint code
        run: echo "Linting repository"

      - name: Run unit tests
        run: echo "Running unit tests"

  build-and-push-ecr-image:
    name: Continuous Delivery
    needs: integration
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v3

      - name: Install Utilities
        run: |
          sudo apt-get update
          sudo apt-get install -y jq unzip
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION }}

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1

      - name: Build, tag, and push image to Amazon ECR
        id: build-image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: ${{ secrets.ECR_REPOSITORY_NAME }}
          IMAGE_TAG: latest
        run: |
          # Build a docker container and
          # push it to ECR so that it can
          # be deployed to ECS.
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          echo "::set-output name=image::$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG"
          
          
  Continuous-Deployment:
    needs: build-and-push-ecr-image
    runs-on: self-hosted
    steps:
      - name: Checkout
        uses: actions/checkout@v3

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION }}

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1
      
      
      - name: Pull latest images
        run: |
         docker pull ${{secrets.AWS_ECR_LOGIN_URI}}/${{ secrets.ECR_REPOSITORY_NAME }}:latest
                
      - name: Run Docker Image to serve users
        run: |
         docker run -d -p 8080:8080 --ipc="host" --name=catdog -e 'AWS_ACCESS_KEY_ID=${{ secrets.AWS_ACCESS_KEY_ID }}' -e 'AWS_SECRET_ACCESS_KEY=${{ secrets.AWS_SECRET_ACCESS_KEY }}' -e 'AWS_REGION=${{ secrets.AWS_REGION }}'  ${{secrets.AWS_ECR_LOGIN_URI}}/${{ secrets.ECR_REPOSITORY_NAME }}:latest
      - name: Clean previous images and containers
        run: |
         docker system prune -f

</pre>

## CI/CD Steps

1)	Push app to GitHub
2)	Open AWS  IAM -> Users -> Create user -> Attach policies directly (add policies) -> Create user -> now open created user -> Create access key -> Retrieve access keys -> Download .csv file
3)	Create ECR repo: Open ECR -> Get started -> private -> write a repo name -> create repository -> copy the URI
4)	Create EC2 machine: Open EC2 -> Launch Instance -> write name of instance -> select OS (take Ubuntu only) -> select any version of Ubuntu -> select instance type (take free tire) -> write key-pair name (login) -> allow HTTPS, HTTPS traffic, and SSH traffic -> setup storage -> Launch instance -> click on ‘view all instances’ to see created EC2 instance status -> once it is running click on ‘Instance ID’ -> Connect -> Connect -> it will launch a terminal -> type below commands to setup Ubuntu system for hosting app
    - sudo apt-get update –y
    - sudo apt-get upgrade
    - curl -fsSL https://get.docker.com -o get-docker.sh
    - sudo sh get-docker.sh
    - sudo usermod -aG docker Ubuntu
    - newgrp docker
    - clear
    - docker --version
5)	Configure EC2 as self-hosted runner:  
    - Open GitHub app repo
        - Settings -> Actions -> Runners -> New self-hosted runner -> select Linux -> See download commands and configure commands -> copy one by one and run it on terminal
        - enter the name of runner: self-hosted
    - see the GitHub app repo to verify the ‘Runners’ working or not
6)	Setup github secrets
    - Settings -> Secret and variables -> Actions -> New repository secret -> 
        - AWS_ACCESS_KEY_ID
        - AWS_SECRET_ACCESS_KEY
        - AWS_REGION
        - AWS_ECR_LOGIN_URI
        - ECR_REPOSITORY_NAME
7)	Any updates/changes in GitHub app files will automatically trigger CI/CD process to update AWS EC2 runner as well.
    - Click on Actions -> click on latest commits (yellow circle) -> a CI/CD flow will be visible
    - To verify, go to ECR and open repo and see the ‘latest’ image tag
8)	Open EC2 instance again
    - Security -> security groups -> edit inbound rules -> add port, rules, etc. -> save rules
    - EC2 Instance -> Copy Public IPv4 address (this is the final link of the web app)
9)	Done!
