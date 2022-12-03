import utils


files = utils.read_files()
boardgames=[]

for file in files:
    rulebook=utils.read_rolebook_pdf(file)
    boardgame=utils.create_boardgame(rulebook)
    boardgames.append(boardgame)


for b in boardgames:
    print(b.name, b.weight, b.rulebook.readability+ b.rulebook.syn_analysis+b.comment_score)

