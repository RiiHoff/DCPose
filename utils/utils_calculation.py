from decimal import Decimal, ROUND_HALF_UP

def rounding(value):

    dvalue = Decimal(str(value))
    r_value = dvalue.quantize(Decimal('0'), rounding=ROUND_HALF_UP)
    return r_value


def round_dp2(value):
    
    dvalue = Decimal(str(value))
    r_value = dvalue.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    return r_value