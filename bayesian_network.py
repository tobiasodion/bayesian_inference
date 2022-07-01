from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

#--------------------RANDOM VARIABLE DESCRIPTION--------------------------#
#L : A prediction from a NLP ML (Natural Language Processing Machine Learning) model that can read the content of the Project 
     #and give a score (signalled) (probability) that this content is copied/plagiated.
#I : Another student(colleague) marks the material as inappropriate/plagiarism.
#S : The Project was suspended before for any bad remarks.
#N : Score (Probability) that the Project should be not considered (not scored)
#R : Score (Probability) that Student should be restricted for this lecture (ratrappage)

#--------------------QUESTION 1--------------------------#
#1. Create the Bayes nets using: bayesNet = BayesianModel()

bayesNet = BayesianNetwork()

#- For all nodes:
#Add the nodes to the bayesian model
bayesNet.add_node('L')
bayesNet.add_node('I')
bayesNet.add_node('S')
bayesNet.add_node('N')
bayesNet.add_node('R')

#- For all edges:
#Add the edges to the bayesian model
bayesNet.add_edge('L', 'N')
bayesNet.add_edge('I', 'N')
bayesNet.add_edge('S', 'N')
bayesNet.add_edge('S', 'R')
bayesNet.add_edge('N', 'R')

#2. Add CPDs to each node, while adding probabilities, we have to give FALSE values first using the function TabularCPD() :

#CPD for node I
i_cpd = TabularCPD(variable='I', 
                    variable_card=2, 
                    values=[[0.79], [0.21]], 
                    state_names={'I' : ['F', 'T']})

#CPD for node L
l_cpd = TabularCPD(variable='L', 
                    variable_card=2, 
                    values=[[0.92], [0.08]], 
                    state_names={'L' : ['F', 'T']})

#CPD for node S
s_cpd = TabularCPD(variable='S', 
                    variable_card=2, 
                    values=[[0.88], [0.12]], 
                    state_names={'S' : ['F', 'T']})

#CPD for node R
r_cpd = TabularCPD(variable='R', 
                    variable_card=2, 
                    values=[[0.95,0.84, 0.92, 0.62],
                            [0.05, 0.16, 0.08, 0.38]],
                    evidence=['N', 'S'],
                    evidence_card=[2, 2],
                    state_names={'R' : ['F', 'T'], 
                                'N' : ['F', 'T'], 
                                'S' : ['F', 'T']})

#CPD for node N
n_cpd = TabularCPD(variable='N', 
                    variable_card=2, 
                    values=[[0.97, 0.83, 0.92, 0.78, 0.27, 0.21, 0.12, 0.08],
                            [0.03, 0.17, 0.08, 0.22, 0.73, 0.79, 0.88, 0.92]],
                    evidence=['L', 'S', 'I'],
                    evidence_card=[2, 2, 2],
                    state_names={'N' : ['F', 'T'], 
                                'L' : ['F', 'T'], 
                                'S' : ['F', 'T'],
                                'I' : ['F', 'T']})

#Extra - print the CPD Tables of each random variable
print('\nI CPD\n')
print(i_cpd)
print('\nL CPD\n')
print(l_cpd)
print('\nS CPD\n')
print(s_cpd)
print('\nR CPD\n')
print(r_cpd)
print('\nN CPD\n')
print(n_cpd)

#Add the CPD tables for each random variable to the Bayes Net
bayesNet.add_cpds(i_cpd,l_cpd,s_cpd,r_cpd,n_cpd)

#3. Check if model is correctly created using bayesNet.check_model()
print('\nBayes Model Check')
print(bayesNet.check_model())
print('\n')

#4. Creating solver that uses variable elimination internally for inference using :
solver = VariableElimination(bayesNet)
#print(solver)


#--------------------QUESTION 2--------------------------#
#Compute proability of « Project should be not considered (not scored)»**
#2. Compute this probability using pgmpy library:
print('\nP(N)\n')
result1 = solver.query(variables=['N'])
print(result1)

#--------------------QUESTION 3--------------------------#
#Compute the probability of « Project should be not considered (no scored) given the ML model signalled it"
#2. Compute this probability using pgmpy library:
print('\nP(N|L=T)\n')
result2 = solver.query(variables=['N'], evidence={'L':'T'})
print(result2)

#--------------------QUESTION 4--------------------------#
#Find (in)dependencies between the variables using the function get_independencies().
print('\nThe following independencies exist between the random variables:\n')
print(bayesNet.get_independencies())