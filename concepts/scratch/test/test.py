d = {"x":"a", "y":"d", "z":"b"}


def order_dict(d:dict) -> dict :
    if len(d) == 0  or len(d) == 1:
        return d
    
    sorted_vals = sorted([val for val in d.values()])

    rev_d = {val:item for item,val in d.items()}


    result = {}
    for val in sorted_vals:
        key = rev_d.get(val)
        if key:
            result[key] = val
    

    return result  


print(order_dict(d))







        
 