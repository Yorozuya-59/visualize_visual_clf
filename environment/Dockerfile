FROM ubuntu:22.04

WORKDIR /home/workdir

# Install Necessary packages
RUN apt update 
RUN apt upgrade -y
RUN apt install -y python3.10 curl python3-pip htop

# For initial setting of Python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 
RUN pip install -U pip 

# Installation of basic libraries
RUN pip install python-dotenv icecream tqdm

# Install Marimo
RUN pip install "marimo[recommend]"
COPY install_gh.sh /usr/local/bin/install_gh.sh
RUN bash /usr/local/bin/install_gh.sh
RUN pip install openai mcp
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
RUN apt install -y nodejs

# Install for MCP
RUN pip install mcp-server-fetch

# Installation of libraries for data analysis
RUN pip install polars

# Installation of libraries for machine learning
RUN pip install scikit-learn 
RUN pip install torch
RUN pip install torcheval torchinfo 
RUN pip install tensorboard

# Installation of libraries for visualization
RUN pip install matplotlib japanize-matplotlib seaborn
