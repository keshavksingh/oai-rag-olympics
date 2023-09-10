from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment
)
from azure.identity import ClientSecretCredential

#Details of AzureML workspace
subscription_id = '<>'
resource_group = 'mlops'
workspace_name = 'amlwscus01'

tenant_id='<>'
client_id='<>'
client_secret="<>"

creds = ClientSecretCredential(tenant_id, client_id, client_secret)
ml_client = MLClient(credential=creds, subscription_id=subscription_id, resource_group_name=resource_group, workspace_name=workspace_name)

endpoint_name = "oairag-endpoint"

endpoint = ManagedOnlineEndpoint(
    name = endpoint_name, 
    description="This is endpoint for Querying Olympics Data on OAI RAG Inferencing for East US Region.",
    auth_mode="key"
)

#model = Model(path="C:/scaled-secure-aml-endpoint/model.pkl")
#env = Environment(conda_file="C:/oai-aml/dev/conda.yml",
#                  image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest")#"ghcr.io/chroma-core/chroma:latest"

chromadb_env = ml_client.environments.get(name='chromadb-env', version="1")
blue_deployment = ManagedOnlineDeployment(
    name="default",
    endpoint_name=endpoint_name,
    #model=model,
    environment=chromadb_env,
    scoring_script="score.py",
    code_path="C:/oai-aml/dev",
    instance_type="Standard_D2as_v4",
    instance_count=1
)

endpoint.traffic = {"default": 100}
endpoint_poller = ml_client.online_endpoints.begin_create_or_update(endpoint)
if endpoint_poller.result():
    print("Endpoint Creation Complete!")
    print(endpoint_poller.result())
    deployment_poller = ml_client.online_deployments.begin_create_or_update(deployment=blue_deployment)
    if deployment_poller.result():
        print("Deployment of Endpoint Complete!")
        print(deployment_poller.result())