class BoardGame:

    def __init__(self, name, comments,weight,expansions_num,rulebook,score):
        self.name = name
        self.comments = comments
        self.BGGweight=weight
        self.expansions_num = expansions_num
        self.rulebook=rulebook
        self.comment_score=score
        self.total_weight=0
