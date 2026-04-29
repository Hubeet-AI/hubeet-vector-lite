## Correr los benchmarks

```
make clean && make hvl-bench && ./hvl-bench
```


## Exportar modelos de embedding
```
pip install torch transformers
python3 tools/export_minilm.py
```

## Configurar el servidor
```
dim = 384
embedding_model_path = "./models/minilm.hvl_model"
```

## Levantar el servidor
```
./hvl-server
```

## Interactuar con el servidor
```
./hvl-cli
> TSEARCH 1 Guarda la respuesta de un agente, y devuelve la URL que hay que llamar para ver el resultado de la ejecución de esta respuesta.
> TSEARCH 2 decentralized finance and cryptographic assets
> TSEARCH 10 decentralized finance and cryptographic assets
>
```


### Visualizar el espacio latente
```
 python3 tools/visualize_dump.py  [PATH_TO_DUMP]
```