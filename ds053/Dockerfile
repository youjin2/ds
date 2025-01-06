FROM tensorflow/tensorflow:2.11.0-gpu

# RUN apt-get update --yes && \
#     apt-get install htop --yes && \
#     apt-get install vim --yes && \
#     apt-get install gcc --yes && \
#     apt-get install g++ --yes && \
#     apt-get install libpq-dev --yes && \
#     apt-get install python-dev --yes

# agree font install license
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections

RUN apt update && \
    apt install graphviz --yes && \
    apt install msttcorefonts --yes && \
    apt install ttf-mscorefonts-installer --yes

RUN echo "alias ll='ls -alF'" >> ~/.bashrc && \
    echo "alias la='ls -A'" >> ~/.bashrc && \
    echo "alias l='ls -CF'" >> ~/.bashrc

COPY requirements.txt /tmp/installs/
RUN pip install -r /tmp/installs/requirements.txt
# COPY example.py /tmp/installs/
# COPY check_gpu.py /tmp/installs/

WORKDIR /opt/project/


