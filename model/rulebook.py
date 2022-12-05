class Rulebook:
    def __init__(self, filename, text, pagenum,num_sent,num_word,num_complex_word,syn_analysis):
        self.filename = filename
        self.text = text
        self.pagenum= pagenum
        self.num_word=num_word
        self.num_sent=num_sent
        self.num_complex_word=num_complex_word
        self.readability=0.4*(self.num_word/self.num_sent + 100*self.num_complex_word/self.num_word) #readability metric
        self.syn_analysis=syn_analysis