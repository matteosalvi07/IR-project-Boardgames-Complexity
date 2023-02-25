import utils
files = utils.read_files()
boardgames=[]

for file in files:
    rulebook=utils.read_rolebook_pdf(file)
    boardgame=utils.create_boardgame(rulebook)
    boardgames.append(boardgame)


for b in boardgames:
    b.total_weight=b.rulebook.readability+ b.rulebook.syn_analysis+b.comment_score
    if b.expansions_num>6:
        b.total_weight=b.total_weight+1
    print(b.name, b.BGGweight, b.total_weight)

