A model to understand the influence of breast cancer cells (BBC) on M1/M2 macrophages. The breast cancer cell number can be interpreted as their proliferation.
For M1/M2 the cell number is less relevant, as they won't proliferate, but are derived from monocytes. Hence we model their activity, based on a relevant secreted cytokine.
For M1 this is TNF-aplha, which can cause cancer cell apoptosis, for M2 this is IL-10, which also enhances BCC activity. The BCC themselves will increase the number of M2 cells,
by recruiting monocytes and polarising them to M2. 


Still unclear but important for continuing:
But if the number of M2 cells remains constant (like in our setup), will BCC still have an effect on M2 activity?
How do we model this activity through an ODE? I'm a bit confused here. We could say that higher cytokine concentration (TNFa/IL10) is higher activity?
How do we incorporate the BCC secreted factors into the model for M2? There's quite a lot of them that enhance BCCs. 
