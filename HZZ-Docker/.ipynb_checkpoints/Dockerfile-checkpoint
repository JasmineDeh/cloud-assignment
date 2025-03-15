# Official Python image as a base
FROM python:3.10

# Working directory in the container
WORKDIR /hzz-analysis

# Copy code into the container
COPY . /hzz-analysis
COPY requirements.txt /hzz-analysis/

# Install pip and upgrade it
RUN python -m pip install --upgrade pip

# Install dependencies for HTTPFileSystem
RUN pip install requests
RUN pip install aiohttp

# Install dependencies from requirements.txt without using cache (used "pip show <package_name>" to reproduce package versions used in local environment)
RUN pip install --no-cache-dir -r requirements.txt

# Run Python script
CMD ["python", "HZZAnalysis.py"]