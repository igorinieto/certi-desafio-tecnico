import matplotlib.pyplot as plt
import numpy as np

"""
Este script serve como um arquivo secundário para armazenar as funções utilizadas no notebook.

"""
def maiuscula(palavra):
    '''
    Função responsável por verificar se a palavra possui letra maiúscula.
    parâmetros:
        - palavra
    return:
        - True: Caso a palavra tenha letra maiúscula
        - False: Caso a palavra não tenha letra maiúscula
    '''
    
    if any(l.isupper() for l in palavra):
        return True
    return False



def plot_imagens_geradas(ger, titulo, dic_labels, print_pred = False, model = None, nrows = 3, ncols = 3, figsize = (12, 12)):
    '''
    Função responsável por plotar as imagens geradas.
    parâmetros:
        - ger: dados gerados
        - titulo: titulo do plot
        - dic_labels: dicionário das labels
    '''
    ger_dados = ger.next()
   
    plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)
    plt.suptitle(titulo, fontsize = 20)
    plt.tight_layout(rect = [0, 0, 1, 0.96], h_pad = 2)
    
    if(print_pred and model):
        pred = np.argmax(model.predict(ger_dados[0]), axis=1)

    for i in range(nrows * ncols):
        plt.subplot(nrows, ncols, i + 1)
        plt.axis(False)
        plt.grid(False)
        
        if(print_pred and pred.any()):
            plt.title(f"True: {dic_labels[ger_dados[1][i]]}\nPredicted: {dic_labels[pred[i]]}")
        else:
            plt.title(dic_labels[ger_dados[1][i]])
        plt.imshow(ger_dados[0][i])
        
        
        
def plot_imagens(imagens, labels_pred, labels_test):
    '''
    Função responsável por plotar array de imagens.
    parâmetros:
        - imagens: array de imagens para serem plotadas
        - labels_pred: array das labels do resultado da classificação das imagens
        - labels_test: array das labels originais das imagens
    '''
    n_cols = 5
    n_rows = len(imagens) // n_cols
    fig = plt.figure(figsize=(15, 15))
 
    for i in range(len(imagens)):
        sp = fig.add_subplot(n_rows, n_cols, i+1)
        plt.axis("off")
        plt.imshow(imagens[i])
        sp.set_title(labels_pred[i], color='black' if labels_pred[i] == labels_test[i] else 'red')
    plt.show()
    
    
    
def labels_palavras(labels):
    """
    Converte labels numéricas em palavras.
    parâmetros: 
        - labels: numpy array de inteiros
    returns: numpy array de palavras
    """
    return np.where(labels == 1, "Cachorro", "Gato")