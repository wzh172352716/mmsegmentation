from mmseg.apis import MMSegInferencer

models = MMSegInferencer.list_models('mmseg')
print(models)