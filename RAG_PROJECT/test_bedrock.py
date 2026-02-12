import boto3
import json 

client = boto3.client("bedrock-runtime", region_name = "us-east-1")

print("Attempting to invoke Titan Embeddings V2...")

try:
    respone = client.invoke_model(

        modelId="amazon.titan-embed-text-v2:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "inputText": "Hello AWS",
            "dimensions" : 1024,
            "normalize": True
        })

    )

    result = json.loads(respone['body'].read())
    print("Success")
    print(f"Vector dims : {len(result['embedding'])}")
    
except Exception as e:
    print(f" Error : {e}")