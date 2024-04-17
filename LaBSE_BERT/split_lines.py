import io

fil9 = 'fil9'
first_par = 'wiki-first-paragraph'
old_fil9 = 'fil9-old'
old_first_par = 'wiki-first-paragraph-old'

curr_in = old_first_par

_input = 'data/' + curr_in
_output = 'split_output/split_output-' + curr_in + '-'

if __name__ == '__main__':
    print("Splitting Lines of ", _input, "\n")
    fin = io.open(_input, 'r', encoding='utf-8', newline='\n', errors='ignore')
    x = fin.readline()
    num = 1

    while x:
        x = x.replace(". ", ".\n").replace("! ", "!\n").replace("? ", "?\n")
        curr_file = open(_output + str(num), "w")
        curr_file.write(x)
        # print(x)
        num += 1
        x = fin.readline()
