from decimal import Decimal

def decim(string,digits):
    return ('{0:.'+str(digits)+'E}').format(Decimal(str(string)))

def spaced_decim(string,digits,width):
    d = decim(string,digits)
    return d + ' '*(width-len(d))

