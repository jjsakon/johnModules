def fake_func(scalp_subs):  
    import sys
    sys.path.append('/home1/john/johnModules')
    #%load_ext autoreload
    #%autoreload 1
    #%aimport PSYC429.Assignment3_Module
    from PSYC429.Assignment3_Module import SubjectStatTable
    
    table1 = SubjectStatTable(scalp_subs)
    a = 5*35
    
    return table1,a
