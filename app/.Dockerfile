FROM python:3.11.4-bookworm

RUN pip3 install dash
RUN pip3 install pandas
RUN pip3 install dash_bootstrap_components
RUN pip3 install dash-bootstrap-components[pandas]
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install scikit-learn
RUN pip3 install shap
RUN pip3 install joblib

COPY ./ /root/app/

CMD tail -f /dev/null