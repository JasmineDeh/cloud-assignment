Cloud Computing coding project: HZZ-Docker

This repository contains the necessary setup to run the HZZ analysis using Docker. Follow the instructions below to get the environment up and running.

Prerequisites:

1. Docker Desktop: Ensure you have Docker Desktop downloaded and installed.

2. Docker Compose: This project uses Docker Compose to manage multi-container Docker applications. Docker Compose comes bundled with Docker Desktop, so there's no need for a separate installation.
Setup Instructions

Follow these steps to run the analysis and access the outputs:

1. Download the HZZ-Docker Folder

Clone or download the HZZ-Docker folder from this repository to your local machine.

3. Change Directory to HZZ-Docker Folder
   
Open a terminal window and navigate to the HZZ-Docker folder:

cd /path/to/HZZ-Docker

3. Build and Start the Docker Containers
   
Run the following command to build and start the Docker containers using Docker Compose:

docker-compose up --build

This will initiate the analysis process in the Docker container. The analysis will run inside the container as specified by the setup.

4. Access the Output Files
   
Once the analysis is complete, you will find the generated data and plots inside the outputs folder within the HZZ-Docker directory.


Make sure Docker Desktop is running before you start the process.
If you encounter any issues during the build or execution, verify that your system meets the Docker Desktop requirements.
To stop the Docker containers, you can run:
docker compose down
