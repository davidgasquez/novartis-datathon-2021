FROM gcr.io/kaggle-images/python:v106

RUN update-alternatives --install /usr/bin/python python /opt/conda/bin/python 10

ARG USERNAME=jupyter
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

ADD "https://raw.githubusercontent.com/microsoft/vscode-dev-containers/main/script-library/common-debian.sh" /tmp/library-scripts/

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive  \
    && apt-get -y install --no-install-recommends curl ca-certificates \
    && bash /tmp/library-scripts/common-debian.sh true jupyter \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN chown 1000:1000 -R /opt/

RUN conda install pylint -y

WORKDIR /workspaces/novartis-datathon-2021
COPY . /workspaces/novartis-datathon-2021

RUN pip install -e .

CMD ["jupyter", "lab",  "--no-browser", "--ip='0.0.0.0'", "--allow-root", "--notebook-dir", "."]
