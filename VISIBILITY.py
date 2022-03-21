def VISIBILITY_GRAPH(text, window_size):
    edge=[]
    if len(text) < window_size:
        window_size = len(text)
    for i in range(len(text) - window_size + 1):
        text_slice = text[i: i + window_size]
        for j in range(window_size - 2):
            edge.append((int(text_slice[j][0]), int(text_slice[j+1][0])))
            edge.append((text_slice[j+1][0],  text_slice[j][0]))
            for k in range(j + 2, window_size):
                slope = (text_slice[k][1] - text_slice[j][1]) / (k - j)
                indicator = 1
                y0 = text_slice[j][1]
                for h in range(j+1, k):
                    y = y0 + slope * (h - j)
                    if y>text_slice[h][1]:
                        indicator = indicator * 1
                    else :
                        indicator = indicator * 0
                if indicator == 1:
                    edge.append((int(text_slice[j][0]), int(text_slice[k][0])))
                    edge.append((text_slice[k][0], text_slice[j][0]))
        edge.append((int(text_slice[window_size - 2][0]), int(text_slice[window_size - 1][0])))
        edge.append((text_slice[window_size - 1][0], text_slice[window_size - 2][0]))

    return  edge



