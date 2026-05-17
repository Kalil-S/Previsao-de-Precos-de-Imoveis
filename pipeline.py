import pandas as pd
import numpy as np
import joblib

def prever_precos(caminho_arquivo_teste):
    """
    Lê o arquivo de teste, aplica todo o pré-processamento salvo
    e retorna as predições em um array Numpy.
    """
    # 1. Carrega os dados de teste
    df_teste = pd.read_csv(caminho_arquivo_teste)
    
    # Salva os IDs caso seja necessário, mas dropa do dataset de inferência
    # (O professor disse que a saída não deve conter a coluna de ID)
    X_teste = df_teste.drop(columns=['Id'], errors='ignore')
    
    # 2. Carrega o pipeline completo treinado (imputadores, encoders, escalonadores e o modelo)
    # Certifique-se de que o 'modelo.pkl' esteja na mesma pasta que este script
    pipeline_carregado = joblib.load('modelo.pkl')
    
    # 3. Realiza a predição
    # OBS: Graças ao TransformedTargetRegressor, o predict JÁ SAI EM DÓLARES (sem precisar de np.expm1 aqui)
    predicoes = pipeline_carregado.predict(X_teste)
    
    # 4. Retorna estritamente um array do Numpy com os valores contínuos (exigência do professor)
    return np.array(predicoes)

# Bloco de execução para o Acompanhamento 2 (Testando se funciona)
if __name__ == "__main__":
    print("Iniciando o Pipeline de Predição...")
    
    # O professor vai usar a base secreta, mas para o teste usamos o teste_publico.csv
    caminho_teste = 'teste_publico.csv' 
    
    try:
        # Chama a função principal
        resultados = prever_precos(caminho_teste)
        
        print("\nPipeline executado com sucesso!")
        print(f"Total de predições geradas: {len(resultados)}")
        print("Primeiros 5 preços previstos:")
        for i, preco in enumerate(resultados[:5]):
            print(f"Casa {i+1}: $ {preco:,.2f}")
            
    except Exception as e:
        print(f"\n[ERRO] O pipeline falhou: {e}")