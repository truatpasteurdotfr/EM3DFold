FROM ghcr.io/prefix-dev/pixi:0.81.0-trixie

RUN apt-get update &&\
 DEBIAN_FRONTEND=noninteractive apt-get -y upgrade &&\
 DEBIAN_FRONTEND=noninteractive apt-get -y install git build-essential cmake  &&\
 DEBIAN_FRONTEND=noninteractive apt-get -y autoremove &&\
 DEBIAN_FRONTEND=noninteractive apt-get -y clean all
RUN date +"%Y-%m-%d-%H%M" > /last_update

RUN  mkdir /app && cd /app && git clone https://github.com/truatpasteurdotfr/EM3DFold.git
WORKDIR /app/EM3DFold
# copy source code, pixi.toml and pixi.lock to the container
#COPY . /app/EM3DFold

RUN	pixi install && \
	pixi run bash flash_attn_whl/install_flash_attn.sh && \
	pixi run pip install .

RUN pixi shell-hook --manifest-path /app/EM3DFold/pixi.toml  > /shell-hook.sh
RUN echo 'exec "$@"' >> /shell-hook.sh

ENTRYPOINT ["/bin/bash", "/shell-hook.sh"]
