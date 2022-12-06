import numpy as np


def hydrate(lines):
    result = {}
    current_model = ''
    mat = []
    for line in lines:
        line_aux = []
        line_split = line.split('; ')
        if line[0] == '=':
            result[int(current_model)] = mat
            mat = []
            continue
        elif line[0] == 'D':
            path_model = line_split[0].split('\\')[-1]
            current_model = path_model.rstrip('.pth')
            line_aux.append(current_model)
            for i in range(1, len(line_split)):
                line_aux.append(line_split[i])
        else:
            for i in range(len(line_split)):
                if i == 0:
                    line_aux.append(line_split[i])
                else:
                    line_aux.append(float(line_split[i]))
        mat.append(line_aux)
    return result


def main():
    # Abre o arquivo
    with open('./v5_02-12-22.csv', 'r') as cartas:
        lines = [line.rstrip('\n') for line in cartas]
    
    # Transforma a lista de linhas em um dicionario por modelo
    dict_file = hydrate(lines)

    # Testes
    # test = [[dict_file[10][0][i], 0, np.Inf] for i in range(1, 50)]
    # for i in range(10, 301, 10):
    #     for j in range(1, 50):
    #         if dict_file[i][j][j] < test[j-1][-1]:
    #             test[j-1]  = [test[j-1][0], i, dict_file[i][j][j]]
    # for i in range(len(test)):
    #     print(test[i])

    # test = {}
    # for i in range(10, 301, 10):
    #     acertos = 0
    #     for j in range(1, 50):
    #         minimo =  np.Inf
    #         for l in   range(1, 50):
    #             if dict_file[i][j][l] < minimo:
    #                 minimo = dict_file[i][j][l]
    #         if minimo == dict_file[i][j][j]:
    #             acertos += 1
    #     test[i] = acertos
    # for i in range(10, 301, 10):
    #     print(i, test[i])


if __name__ == '__main__':
    main()
