"""Utility fucntions for using ollama"""

async def pull_models(models: List[str]):
    """uses ollama to pull one or more models"""

    missing_models = await check_for_missing_models(models)

    print(f"models already installed: {[set(models) - set(missing_models)]}")

    for model in missing_models:
        await pull_model(model)

    if missing_models is not None:
        print(f"All models have been pulled! ({' '.join(missing_models)})\n\n")


async def check_for_missing_models(models: List[str]) -> List[str]:
    """Checks to see what models are already present, returns a list of the missing models"""

    installed_models = (await ollama_client.list()).get("models")
    if installed_models is not None:
        installed_models = [model.get("name") for model in installed_models]

    return [model for model in models if model not in installed_models]
