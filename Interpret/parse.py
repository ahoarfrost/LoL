#! usr/bin/python
import sys

# Classes to store the output of a topmatch run
class Record():
    def __init__(self,Version, Query, Target, Target_Len):
        self.Version = Version
        self.Query = Query
        self.Target = Target
        self.Target_Len = Target_Len
        self.rank = {}

    def add_align(self, align):
        self.rank[ int(align[0]) ] = Alignment(*align)

    def __repr__(self):    
        temp = 'TopMatch Version: ' + str(self.Version) + '\n'
        temp += 'Query: ' + str(self.Query) + '\n'
        temp += 'Target: ' + str(self.Target) + '\n'
        temp += '#' * 55 + '\n'
        for rank in self.rank.keys() :
            temp += str(self.rank[rank]) + '\n'    
        return temp
    # From https://topmatch.services.came.sbg.ac.at/help/tm_help.html    
    #T    Type of alignment. Basic structure alignments are denoted by 'b', composite alignments are denoted by 'c', and sequence alignments are denoted by 'q'.
    #R    Rank of alignment.
    #S    Measure of structural similarity based on Gaussian functions (see Sippl & Wiederstein (2012)). If the structurally equivalent parts in query and target match perfectly, S is equal to L. With increasing spatial deviation of the aligned residues, S approaches 0.
    #Sq    Query cover based on similarity score, expressed in percent (= 100 x S/Qn, where Qn is the length of the query sequence).
    #St    Target cover based on similarity score, expressed in percent(= 100 x S/Tn, where Tn is the length of the target sequence).
    #L    Number of residue pairs that are structurally equivalent(= alignment length).
    #Lq    Query cover based on alignment length, expressed in percent(= 100 x L/Qn, where Qn is the length of the query sequence).
    #Lt    Target cover based on alignment length, expressed in percent(= 100 x L/Tn, where Tn is the length of the target sequence).
    #Sr    Typical distance error. For details on the construction of this per-residue measure of structural similarity, see Sippl & Wiederstein (2012).
    #Er    Root-mean-square error of superposition in Angstrom, calculated using all structurally equivalent C-alpha atoms.
    #Is    Sequence identity of query and target in the equivalent regions, expressed in percent.
    #P    Number of permutations in the alignment.
class Alignment():
    def __init__(self, RANK, STRSCR, QC, TC, LEN, RMS, PRM, ANG, BLKS, SI):
        self.RANK = RANK
        self.STRSCR = STRSCR
        self.QC = QC
        self.TC = TC
        self.LEN = int(LEN)
        self.RMS = RMS
        self.PRM = PRM
        self.ANG = ANG
        self.BLKS = BLKS
        self.SI = SI
        self.block = { }

    def Beg(self):
        return min([int(self.block[i].Tbeg) for i in self.block.keys() ])

    def End(self):
        return max([int(self.block[i].Tend) for i in self.block.keys() ])

    def TileLen(self):    
        Q_LEN = 0 
        T_LEN = 0
        for rank in self.block.keys() :
            block = self.block[rank]
            Q_LEN += int(block.Qend) - int(block.Qbeg)
            T_LEN += int(block.Tend) - int(block.Tbeg) 
        return T_LEN

    def TileOverlap(self, other):
        assert type(self) == type(other)
        for rankA in self.block.keys() :
            start = int(self.block[rankA].Tbeg) 
            stop = int(self.block[rankA].Tend)
            for rankB in other.block.keys() :
                startB = int(other.block[rankB].Tbeg) 
                stopB = int(other.block[rankB].Tend) 
                #print(start,stop, startB, stopB)

                if startB >= start and startB <= stop:
                    return True
                if stopB >= start and stopB <= stop:
                    return True 
                if startB <= start and stopB >= stop:
                    return True
                if start <= startB and stop >= stopB:
                    return True        
        return False

    def add_blk(self, blk):
        self.block[ int(blk[0]) ] = Block(*blk)
    
    def __repr__(self): 
        temp = 'Align ' + str(self.RANK) + ':\n\t'
        temp += 'RANK:' + str(self.RANK) + '\n\t'
        temp += 'TileLen:' + str(self.TileLen()) + '\n\t'

        temp += 'STRSCR:' + str(self.STRSCR) + '\n\t'
        temp += 'QC:' + str(self.QC) + '\n\t'
        temp += 'TC:' + str(self.TC) + '\n\t'
        temp += 'LEN:' + str(self.LEN) + '\n\t'
        temp += 'RMS:' + str(self.RMS) + '\n\t'
        temp += 'PRM:' + str(self.PRM) + '\n\t'
        temp += 'ANG:' + str(self.ANG) + '\n\t'
        temp += 'BLKS:' + str(self.BLKS) + '\n\t'
        temp += 'SI:' + str(self.SI) + '\n\t'
        temp += 'Blocks: \n' 
        temp += '\tblock\tsize\tdiag\tSI%\tQ-beg\tQ-end\tT-beg\tT-end\tl-rms\tg-rms\ttilt\tshift\tsh\n'
        for rank in self.block.keys() :
            temp += '\t' + str(self.block[rank]) + '\n'    
        return temp

class Block():
    def __init__(self, block, size, diag, SI, Ql, Qbeg, Ql2, Qend, Tl, Tbeg, Tl2, Tend, lrms, grms, tilt, shift, sh = 0):
       self.block = block 
       self.size = size
       self.diag = diag
       self.SI = SI
       self.Qbeg = Qbeg
       self.Qend = Qend
       self.Tbeg = Tbeg
       self.Tend = Tend
       self.lrms = lrms
       self.grms = grms
       self.tilt = tilt
       self.shift = shift
       self.sh = sh     
    def __repr__(self):
       temp =  str(self.block) 
       temp += '\t' + str(self.size )
       temp += '\t' + str(self.diag )
       temp += '\t' + str(self.SI)
       temp += '\t' + str(self.Qbeg )
       temp += '\t' + str(self.Qend )
       temp += '\t' + str(self.Tbeg )
       temp += '\t' + str(self.Tend )
       temp += '\t' + str(self.lrms )
       temp += '\t' + str(self.grms )
       temp += '\t' + str(self.tilt )
       temp += '\t' + str(self.shift)
       temp += '\t' + str(self.sh  )  
       return temp 

def parse_topmatch(file_path):
    handle = open(file_path,'r')
    for line in handle:
        # V 7.5
        if 'TopMatch' in line and '***' not in line:
            version = line.split('-')[1].split(':')[0]
            #print(version)
            break
        # V 7.3
        if 'TopMatch-' in line:
            version = line.split('-')[1].split(':')[0]
            #print(version)
            break
    for line in handle:
        # V 7.5
        if 'fold alignment(s) of' in line :
            t = line.rstrip('\n').split(' of ')[1].split('and')
            query = t[0]
            target = t[1]
            N_of_ranks = line.split()[0] 
            target_len = target.split()[1].rstrip('()')

            #print(query,target,N_of_ranks)
            break
        # V 7.3
        if 'Alignments of' in line :
            t = line.rstrip('\n').split()
            query = t[3]
            target = t[6]
            target_len = int(t[7].strip('()'))
            N_of_ranks = line.split()[0] 
            #print(query,target,N_of_ranks)
            break

    try:        
        # Initialize record
        record = Record(version, query, target, target_len)
    except UnboundLocalError:
        print('Parse error', file_path)
        sys.exit(0)

    if version == '7.5':    
        # Now parse the alignment information from the record :
        ###########################################################################
        flag = False 
        for line in handle:
            if  "RANK  STRSCR   QC   TC    LEN    RMS  PRM    ANG" in line:
                flag = True    
            if "---------------------------------------------" in line and flag:
                break

        for line in handle:  
            # If i finish this outuput block, stop.    
            if "-------------------------------------------" in line:
                break
            # Line is expected to contain:
            # RANK, STRSCR, QC, TC, LEN, RMS, PRM, ANG, BLKS, SI
            line = line.rstrip('\n').split()
            record.add_align(tuple(line))
        ###########################################################################

        # Parse blocks:
        ###########################################################################
        while True:
            for line in handle:  
                if "Blocks of alignment" in line:
                    rank = int(line.split()[-1:][0]) 
                    break
            flag = False
            for line in handle:  
                if " block  size  diag    SI%    Q-beg    Q-end    T-beg    T-end" in line:
                    flag =  True
                if "-------------------------------------------" in line and flag:
                    break

            for line in handle:  
                if "-------------------------------------------" in line:
                    break
                ### ver test_out.tm
                line = line.rstrip('\n').split()
                record.rank[rank].add_blk(tuple(line))
            try:
                line = next(handle)
            except StopIteration:
                break    
        return record
    elif version == '7.3': 
        #print(version)   
         # Now parse the alignment iformation from the record :
        ###########################################################################
        c = 0 
        for line in handle:
            if  "T   R      S   Sq   St      L   Lq   Lt     Sr     Er     Is    P" in line:
                c+= 1    
            if "---------------------------------------------" in line and c > 1:
                break

        for line in handle:  
            # If i finish this outuput block, stop.    
            if "-------------------------------------------" in line:
                break
            # Line is expected to contain:
            # RANK, STRSCR, QC, TC, LEN, RMS, PRM, ANG, BLKS, SI
            # But in V 7.3 it contains 
            # "T   R      S   Sq   St      L   Lq   Lt     Sr     Er     Is    P"
            # So we remove T and P, RANK AND LEN WORK; CHECK ALL ELSE
            line = line.rstrip('\n').split()
            record.add_align(tuple(line[1:-1]))
        ###########################################################################
        
        # Parse blocks:
        ###########################################################################
        while True:
            for line in handle:  
                if "Blocks of alignment" in line:
                    rank = int(line.split()[-1:][0]) 
                    break
            flag = False
            for line in handle:  
                
                if "block   Q-start   Q-end   T-start   T-end size" in line:
                    flag =  True
                if "-------------------------------------------" in line and flag:
                    break

            for line in handle:  
                if "-------------------------------------------" in line:
                    break
                ### ver test_out.tm
                line = line.rstrip('\n').split()
                #block, size, diag, SI, Ql, Qbeg, Ql2, Qend, Tl, Tbeg, Tl2, Tend, lrms, grms, tilt, shift, sh = 0
                block = line[0]
                Ql = line[1]
                Qbeg = line[2]
                Ql2 = line[3]
                Qend = line[4]
                Tl = line[5]
                Tbeg = line[6]
                Tl2 = line[7]
                Tend = line[8]
                size = line[9]
                # non existent on V 7.3 block
                diag,SI,lrms, grms, tilt, shift = 0,0,0,0,0,0
                record.rank[rank].add_blk((block, size, diag, SI, Ql, Qbeg, Ql2, Qend, Tl, Tbeg, Tl2, Tend, lrms, grms, tilt, shift))
            try:
                line = next(handle)
            except StopIteration:
                break    
        return record
        return record
    else:
        print(version, 'cant be parsed')    

#if __name__ == "__main__":
    #record = parse_topmatch('/home/ariel/TopTile/1NOR_A/tm_out/1nor_A_3_8_out.tm')
    #print(record)
#    aln1 = record.rank[1]
#    aln2 = record.rank[2]
#    print('Overlap? : 'aln1.TileOverlap(aln2) )
    