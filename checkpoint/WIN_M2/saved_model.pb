î
Ø
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8·¥
¬
*Adam/dueling_deep_q_network/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/dueling_deep_q_network/dense_3/bias/v
¥
>Adam/dueling_deep_q_network/dense_3/bias/v/Read/ReadVariableOpReadVariableOp*Adam/dueling_deep_q_network/dense_3/bias/v*
_output_shapes
:*
dtype0
µ
,Adam/dueling_deep_q_network/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*=
shared_name.,Adam/dueling_deep_q_network/dense_3/kernel/v
®
@Adam/dueling_deep_q_network/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/dueling_deep_q_network/dense_3/kernel/v*
_output_shapes
:	*
dtype0
¬
*Adam/dueling_deep_q_network/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/dueling_deep_q_network/dense_2/bias/v
¥
>Adam/dueling_deep_q_network/dense_2/bias/v/Read/ReadVariableOpReadVariableOp*Adam/dueling_deep_q_network/dense_2/bias/v*
_output_shapes
:*
dtype0
µ
,Adam/dueling_deep_q_network/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*=
shared_name.,Adam/dueling_deep_q_network/dense_2/kernel/v
®
@Adam/dueling_deep_q_network/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/dueling_deep_q_network/dense_2/kernel/v*
_output_shapes
:	*
dtype0
­
*Adam/dueling_deep_q_network/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/dueling_deep_q_network/dense_1/bias/v
¦
>Adam/dueling_deep_q_network/dense_1/bias/v/Read/ReadVariableOpReadVariableOp*Adam/dueling_deep_q_network/dense_1/bias/v*
_output_shapes	
:*
dtype0
¶
,Adam/dueling_deep_q_network/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,Adam/dueling_deep_q_network/dense_1/kernel/v
¯
@Adam/dueling_deep_q_network/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/dueling_deep_q_network/dense_1/kernel/v* 
_output_shapes
:
*
dtype0
©
(Adam/dueling_deep_q_network/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/dueling_deep_q_network/dense/bias/v
¢
<Adam/dueling_deep_q_network/dense/bias/v/Read/ReadVariableOpReadVariableOp(Adam/dueling_deep_q_network/dense/bias/v*
_output_shapes	
:*
dtype0
±
*Adam/dueling_deep_q_network/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*;
shared_name,*Adam/dueling_deep_q_network/dense/kernel/v
ª
>Adam/dueling_deep_q_network/dense/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/dueling_deep_q_network/dense/kernel/v*
_output_shapes
:	*
dtype0
¬
*Adam/dueling_deep_q_network/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/dueling_deep_q_network/dense_3/bias/m
¥
>Adam/dueling_deep_q_network/dense_3/bias/m/Read/ReadVariableOpReadVariableOp*Adam/dueling_deep_q_network/dense_3/bias/m*
_output_shapes
:*
dtype0
µ
,Adam/dueling_deep_q_network/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*=
shared_name.,Adam/dueling_deep_q_network/dense_3/kernel/m
®
@Adam/dueling_deep_q_network/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/dueling_deep_q_network/dense_3/kernel/m*
_output_shapes
:	*
dtype0
¬
*Adam/dueling_deep_q_network/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/dueling_deep_q_network/dense_2/bias/m
¥
>Adam/dueling_deep_q_network/dense_2/bias/m/Read/ReadVariableOpReadVariableOp*Adam/dueling_deep_q_network/dense_2/bias/m*
_output_shapes
:*
dtype0
µ
,Adam/dueling_deep_q_network/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*=
shared_name.,Adam/dueling_deep_q_network/dense_2/kernel/m
®
@Adam/dueling_deep_q_network/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/dueling_deep_q_network/dense_2/kernel/m*
_output_shapes
:	*
dtype0
­
*Adam/dueling_deep_q_network/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/dueling_deep_q_network/dense_1/bias/m
¦
>Adam/dueling_deep_q_network/dense_1/bias/m/Read/ReadVariableOpReadVariableOp*Adam/dueling_deep_q_network/dense_1/bias/m*
_output_shapes	
:*
dtype0
¶
,Adam/dueling_deep_q_network/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,Adam/dueling_deep_q_network/dense_1/kernel/m
¯
@Adam/dueling_deep_q_network/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/dueling_deep_q_network/dense_1/kernel/m* 
_output_shapes
:
*
dtype0
©
(Adam/dueling_deep_q_network/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/dueling_deep_q_network/dense/bias/m
¢
<Adam/dueling_deep_q_network/dense/bias/m/Read/ReadVariableOpReadVariableOp(Adam/dueling_deep_q_network/dense/bias/m*
_output_shapes	
:*
dtype0
±
*Adam/dueling_deep_q_network/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*;
shared_name,*Adam/dueling_deep_q_network/dense/kernel/m
ª
>Adam/dueling_deep_q_network/dense/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/dueling_deep_q_network/dense/kernel/m*
_output_shapes
:	*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

#dueling_deep_q_network/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#dueling_deep_q_network/dense_3/bias

7dueling_deep_q_network/dense_3/bias/Read/ReadVariableOpReadVariableOp#dueling_deep_q_network/dense_3/bias*
_output_shapes
:*
dtype0
§
%dueling_deep_q_network/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*6
shared_name'%dueling_deep_q_network/dense_3/kernel
 
9dueling_deep_q_network/dense_3/kernel/Read/ReadVariableOpReadVariableOp%dueling_deep_q_network/dense_3/kernel*
_output_shapes
:	*
dtype0

#dueling_deep_q_network/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#dueling_deep_q_network/dense_2/bias

7dueling_deep_q_network/dense_2/bias/Read/ReadVariableOpReadVariableOp#dueling_deep_q_network/dense_2/bias*
_output_shapes
:*
dtype0
§
%dueling_deep_q_network/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*6
shared_name'%dueling_deep_q_network/dense_2/kernel
 
9dueling_deep_q_network/dense_2/kernel/Read/ReadVariableOpReadVariableOp%dueling_deep_q_network/dense_2/kernel*
_output_shapes
:	*
dtype0

#dueling_deep_q_network/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#dueling_deep_q_network/dense_1/bias

7dueling_deep_q_network/dense_1/bias/Read/ReadVariableOpReadVariableOp#dueling_deep_q_network/dense_1/bias*
_output_shapes	
:*
dtype0
¨
%dueling_deep_q_network/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%dueling_deep_q_network/dense_1/kernel
¡
9dueling_deep_q_network/dense_1/kernel/Read/ReadVariableOpReadVariableOp%dueling_deep_q_network/dense_1/kernel* 
_output_shapes
:
*
dtype0

!dueling_deep_q_network/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!dueling_deep_q_network/dense/bias

5dueling_deep_q_network/dense/bias/Read/ReadVariableOpReadVariableOp!dueling_deep_q_network/dense/bias*
_output_shapes	
:*
dtype0
£
#dueling_deep_q_network/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*4
shared_name%#dueling_deep_q_network/dense/kernel

7dueling_deep_q_network/dense/kernel/Read/ReadVariableOpReadVariableOp#dueling_deep_q_network/dense/kernel*
_output_shapes
:	*
dtype0
y
serving_default_args_0Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ñ
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0#dueling_deep_q_network/dense/kernel!dueling_deep_q_network/dense/bias%dueling_deep_q_network/dense_1/kernel#dueling_deep_q_network/dense_1/bias%dueling_deep_q_network/dense_2/kernel#dueling_deep_q_network/dense_2/bias%dueling_deep_q_network/dense_3/kernel#dueling_deep_q_network/dense_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *1
f,R*
(__inference_signature_wrapper_1811993794

NoOpNoOp
ê7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¥7
value7B7 B7

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

dense1

	dense2

V
A
	optimizer

signatures
#_self_saveable_object_factories*
<
0
1
2
3
4
5
6
7*
<
0
1
2
3
4
5
6
7*
* 
°
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
Ë
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

kernel
bias
#&_self_saveable_object_factories*
Ë
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

kernel
bias
#-_self_saveable_object_factories*
Ë
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

kernel
bias
#4_self_saveable_object_factories*
Ë
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

kernel
bias
#;_self_saveable_object_factories*
Ô
<iter

=beta_1

>beta_2
	?decay
@learning_ratem[m\m]m^m_m`mambvcvdvevfvgvhvivj*

Aserving_default* 
* 
c]
VARIABLE_VALUE#dueling_deep_q_network/dense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!dueling_deep_q_network/dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%dueling_deep_q_network/dense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#dueling_deep_q_network/dense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%dueling_deep_q_network/dense_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#dueling_deep_q_network/dense_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%dueling_deep_q_network/dense_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#dueling_deep_q_network/dense_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
	1

2
3*

B0*
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 

Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*
* 
* 
* 

0
1*

0
1*
* 

Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*
* 
* 
* 

0
1*

0
1*
* 

Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*
* 
* 
* 

0
1*

0
1*
* 

Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
8
W	variables
X	keras_api
	Ytotal
	Zcount*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Y0
Z1*

W	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/dueling_deep_q_network/dense/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/dueling_deep_q_network/dense/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/dueling_deep_q_network/dense_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/dueling_deep_q_network/dense_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/dueling_deep_q_network/dense_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/dueling_deep_q_network/dense_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/dueling_deep_q_network/dense_3/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/dueling_deep_q_network/dense_3/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/dueling_deep_q_network/dense/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/dueling_deep_q_network/dense/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/dueling_deep_q_network/dense_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/dueling_deep_q_network/dense_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/dueling_deep_q_network/dense_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/dueling_deep_q_network/dense_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/dueling_deep_q_network/dense_3/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/dueling_deep_q_network/dense_3/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ñ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename7dueling_deep_q_network/dense/kernel/Read/ReadVariableOp5dueling_deep_q_network/dense/bias/Read/ReadVariableOp9dueling_deep_q_network/dense_1/kernel/Read/ReadVariableOp7dueling_deep_q_network/dense_1/bias/Read/ReadVariableOp9dueling_deep_q_network/dense_2/kernel/Read/ReadVariableOp7dueling_deep_q_network/dense_2/bias/Read/ReadVariableOp9dueling_deep_q_network/dense_3/kernel/Read/ReadVariableOp7dueling_deep_q_network/dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp>Adam/dueling_deep_q_network/dense/kernel/m/Read/ReadVariableOp<Adam/dueling_deep_q_network/dense/bias/m/Read/ReadVariableOp@Adam/dueling_deep_q_network/dense_1/kernel/m/Read/ReadVariableOp>Adam/dueling_deep_q_network/dense_1/bias/m/Read/ReadVariableOp@Adam/dueling_deep_q_network/dense_2/kernel/m/Read/ReadVariableOp>Adam/dueling_deep_q_network/dense_2/bias/m/Read/ReadVariableOp@Adam/dueling_deep_q_network/dense_3/kernel/m/Read/ReadVariableOp>Adam/dueling_deep_q_network/dense_3/bias/m/Read/ReadVariableOp>Adam/dueling_deep_q_network/dense/kernel/v/Read/ReadVariableOp<Adam/dueling_deep_q_network/dense/bias/v/Read/ReadVariableOp@Adam/dueling_deep_q_network/dense_1/kernel/v/Read/ReadVariableOp>Adam/dueling_deep_q_network/dense_1/bias/v/Read/ReadVariableOp@Adam/dueling_deep_q_network/dense_2/kernel/v/Read/ReadVariableOp>Adam/dueling_deep_q_network/dense_2/bias/v/Read/ReadVariableOp@Adam/dueling_deep_q_network/dense_3/kernel/v/Read/ReadVariableOp>Adam/dueling_deep_q_network/dense_3/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_save_1811993910

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename#dueling_deep_q_network/dense/kernel!dueling_deep_q_network/dense/bias%dueling_deep_q_network/dense_1/kernel#dueling_deep_q_network/dense_1/bias%dueling_deep_q_network/dense_2/kernel#dueling_deep_q_network/dense_2/bias%dueling_deep_q_network/dense_3/kernel#dueling_deep_q_network/dense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount*Adam/dueling_deep_q_network/dense/kernel/m(Adam/dueling_deep_q_network/dense/bias/m,Adam/dueling_deep_q_network/dense_1/kernel/m*Adam/dueling_deep_q_network/dense_1/bias/m,Adam/dueling_deep_q_network/dense_2/kernel/m*Adam/dueling_deep_q_network/dense_2/bias/m,Adam/dueling_deep_q_network/dense_3/kernel/m*Adam/dueling_deep_q_network/dense_3/bias/m*Adam/dueling_deep_q_network/dense/kernel/v(Adam/dueling_deep_q_network/dense/bias/v,Adam/dueling_deep_q_network/dense_1/kernel/v*Adam/dueling_deep_q_network/dense_1/bias/v,Adam/dueling_deep_q_network/dense_2/kernel/v*Adam/dueling_deep_q_network/dense_2/bias/v,Adam/dueling_deep_q_network/dense_3/kernel/v*Adam/dueling_deep_q_network/dense_3/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference__traced_restore_1811994013á
Ç	
ò
@__inference_dense_3_layer_call_and_return_conditional_losses_678

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥	
¾
(__inference_signature_wrapper_1811993794

args_0
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference__wrapped_model_1811993765o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0
º
í
&__inference__traced_restore_1811994013
file_prefixG
4assignvariableop_dueling_deep_q_network_dense_kernel:	C
4assignvariableop_1_dueling_deep_q_network_dense_bias:	L
8assignvariableop_2_dueling_deep_q_network_dense_1_kernel:
E
6assignvariableop_3_dueling_deep_q_network_dense_1_bias:	K
8assignvariableop_4_dueling_deep_q_network_dense_2_kernel:	D
6assignvariableop_5_dueling_deep_q_network_dense_2_bias:K
8assignvariableop_6_dueling_deep_q_network_dense_3_kernel:	D
6assignvariableop_7_dueling_deep_q_network_dense_3_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: Q
>assignvariableop_15_adam_dueling_deep_q_network_dense_kernel_m:	K
<assignvariableop_16_adam_dueling_deep_q_network_dense_bias_m:	T
@assignvariableop_17_adam_dueling_deep_q_network_dense_1_kernel_m:
M
>assignvariableop_18_adam_dueling_deep_q_network_dense_1_bias_m:	S
@assignvariableop_19_adam_dueling_deep_q_network_dense_2_kernel_m:	L
>assignvariableop_20_adam_dueling_deep_q_network_dense_2_bias_m:S
@assignvariableop_21_adam_dueling_deep_q_network_dense_3_kernel_m:	L
>assignvariableop_22_adam_dueling_deep_q_network_dense_3_bias_m:Q
>assignvariableop_23_adam_dueling_deep_q_network_dense_kernel_v:	K
<assignvariableop_24_adam_dueling_deep_q_network_dense_bias_v:	T
@assignvariableop_25_adam_dueling_deep_q_network_dense_1_kernel_v:
M
>assignvariableop_26_adam_dueling_deep_q_network_dense_1_bias_v:	S
@assignvariableop_27_adam_dueling_deep_q_network_dense_2_kernel_v:	L
>assignvariableop_28_adam_dueling_deep_q_network_dense_2_bias_v:S
@assignvariableop_29_adam_dueling_deep_q_network_dense_3_kernel_v:	L
>assignvariableop_30_adam_dueling_deep_q_network_dense_3_bias_v:
identity_32¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Þ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueúB÷ B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH°
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Á
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::*.
dtypes$
"2 	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp4assignvariableop_dueling_deep_q_network_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_1AssignVariableOp4assignvariableop_1_dueling_deep_q_network_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_2AssignVariableOp8assignvariableop_2_dueling_deep_q_network_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_3AssignVariableOp6assignvariableop_3_dueling_deep_q_network_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_4AssignVariableOp8assignvariableop_4_dueling_deep_q_network_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_5AssignVariableOp6assignvariableop_5_dueling_deep_q_network_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_6AssignVariableOp8assignvariableop_6_dueling_deep_q_network_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_7AssignVariableOp6assignvariableop_7_dueling_deep_q_network_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_15AssignVariableOp>assignvariableop_15_adam_dueling_deep_q_network_dense_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_16AssignVariableOp<assignvariableop_16_adam_dueling_deep_q_network_dense_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_17AssignVariableOp@assignvariableop_17_adam_dueling_deep_q_network_dense_1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_18AssignVariableOp>assignvariableop_18_adam_dueling_deep_q_network_dense_1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_19AssignVariableOp@assignvariableop_19_adam_dueling_deep_q_network_dense_2_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_20AssignVariableOp>assignvariableop_20_adam_dueling_deep_q_network_dense_2_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_21AssignVariableOp@assignvariableop_21_adam_dueling_deep_q_network_dense_3_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_22AssignVariableOp>assignvariableop_22_adam_dueling_deep_q_network_dense_3_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_23AssignVariableOp>assignvariableop_23_adam_dueling_deep_q_network_dense_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_24AssignVariableOp<assignvariableop_24_adam_dueling_deep_q_network_dense_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_25AssignVariableOp@assignvariableop_25_adam_dueling_deep_q_network_dense_1_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_26AssignVariableOp>assignvariableop_26_adam_dueling_deep_q_network_dense_1_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_27AssignVariableOp@assignvariableop_27_adam_dueling_deep_q_network_dense_2_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_28AssignVariableOp>assignvariableop_28_adam_dueling_deep_q_network_dense_2_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_29AssignVariableOp@assignvariableop_29_adam_dueling_deep_q_network_dense_3_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_30AssignVariableOp>assignvariableop_30_adam_dueling_deep_q_network_dense_3_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ù
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: æ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ç	
ò
@__inference_dense_2_layer_call_and_return_conditional_losses_399

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´

%__inference__wrapped_model_1811993765

args_04
!dueling_deep_q_network_1811993747:	0
!dueling_deep_q_network_1811993749:	5
!dueling_deep_q_network_1811993751:
0
!dueling_deep_q_network_1811993753:	4
!dueling_deep_q_network_1811993755:	/
!dueling_deep_q_network_1811993757:4
!dueling_deep_q_network_1811993759:	/
!dueling_deep_q_network_1811993761:
identity¢.dueling_deep_q_network/StatefulPartitionedCallä
.dueling_deep_q_network/StatefulPartitionedCallStatefulPartitionedCallargs_0!dueling_deep_q_network_1811993747!dueling_deep_q_network_1811993749!dueling_deep_q_network_1811993751!dueling_deep_q_network_1811993753!dueling_deep_q_network_1811993755!dueling_deep_q_network_1811993757!dueling_deep_q_network_1811993759!dueling_deep_q_network_1811993761*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *1
f,R*
(__inference_restored_function_body_19575
IdentityIdentity7dueling_deep_q_network/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp/^dueling_deep_q_network/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2`
.dueling_deep_q_network/StatefulPartitionedCall.dueling_deep_q_network/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0
­	
½
(__inference_restored_function_body_19575	
state
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_dueling_deep_q_network_layer_call_and_return_conditional_losses_535o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namestate
I
ø
#__inference__traced_save_1811993910
file_prefixB
>savev2_dueling_deep_q_network_dense_kernel_read_readvariableop@
<savev2_dueling_deep_q_network_dense_bias_read_readvariableopD
@savev2_dueling_deep_q_network_dense_1_kernel_read_readvariableopB
>savev2_dueling_deep_q_network_dense_1_bias_read_readvariableopD
@savev2_dueling_deep_q_network_dense_2_kernel_read_readvariableopB
>savev2_dueling_deep_q_network_dense_2_bias_read_readvariableopD
@savev2_dueling_deep_q_network_dense_3_kernel_read_readvariableopB
>savev2_dueling_deep_q_network_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopI
Esavev2_adam_dueling_deep_q_network_dense_kernel_m_read_readvariableopG
Csavev2_adam_dueling_deep_q_network_dense_bias_m_read_readvariableopK
Gsavev2_adam_dueling_deep_q_network_dense_1_kernel_m_read_readvariableopI
Esavev2_adam_dueling_deep_q_network_dense_1_bias_m_read_readvariableopK
Gsavev2_adam_dueling_deep_q_network_dense_2_kernel_m_read_readvariableopI
Esavev2_adam_dueling_deep_q_network_dense_2_bias_m_read_readvariableopK
Gsavev2_adam_dueling_deep_q_network_dense_3_kernel_m_read_readvariableopI
Esavev2_adam_dueling_deep_q_network_dense_3_bias_m_read_readvariableopI
Esavev2_adam_dueling_deep_q_network_dense_kernel_v_read_readvariableopG
Csavev2_adam_dueling_deep_q_network_dense_bias_v_read_readvariableopK
Gsavev2_adam_dueling_deep_q_network_dense_1_kernel_v_read_readvariableopI
Esavev2_adam_dueling_deep_q_network_dense_1_bias_v_read_readvariableopK
Gsavev2_adam_dueling_deep_q_network_dense_2_kernel_v_read_readvariableopI
Esavev2_adam_dueling_deep_q_network_dense_2_bias_v_read_readvariableopK
Gsavev2_adam_dueling_deep_q_network_dense_3_kernel_v_read_readvariableopI
Esavev2_adam_dueling_deep_q_network_dense_3_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Û
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueúB÷ B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH­
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Û
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0>savev2_dueling_deep_q_network_dense_kernel_read_readvariableop<savev2_dueling_deep_q_network_dense_bias_read_readvariableop@savev2_dueling_deep_q_network_dense_1_kernel_read_readvariableop>savev2_dueling_deep_q_network_dense_1_bias_read_readvariableop@savev2_dueling_deep_q_network_dense_2_kernel_read_readvariableop>savev2_dueling_deep_q_network_dense_2_bias_read_readvariableop@savev2_dueling_deep_q_network_dense_3_kernel_read_readvariableop>savev2_dueling_deep_q_network_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopEsavev2_adam_dueling_deep_q_network_dense_kernel_m_read_readvariableopCsavev2_adam_dueling_deep_q_network_dense_bias_m_read_readvariableopGsavev2_adam_dueling_deep_q_network_dense_1_kernel_m_read_readvariableopEsavev2_adam_dueling_deep_q_network_dense_1_bias_m_read_readvariableopGsavev2_adam_dueling_deep_q_network_dense_2_kernel_m_read_readvariableopEsavev2_adam_dueling_deep_q_network_dense_2_bias_m_read_readvariableopGsavev2_adam_dueling_deep_q_network_dense_3_kernel_m_read_readvariableopEsavev2_adam_dueling_deep_q_network_dense_3_bias_m_read_readvariableopEsavev2_adam_dueling_deep_q_network_dense_kernel_v_read_readvariableopCsavev2_adam_dueling_deep_q_network_dense_bias_v_read_readvariableopGsavev2_adam_dueling_deep_q_network_dense_1_kernel_v_read_readvariableopEsavev2_adam_dueling_deep_q_network_dense_1_bias_v_read_readvariableopGsavev2_adam_dueling_deep_q_network_dense_2_kernel_v_read_readvariableopEsavev2_adam_dueling_deep_q_network_dense_2_bias_v_read_readvariableopGsavev2_adam_dueling_deep_q_network_dense_3_kernel_v_read_readvariableopEsavev2_adam_dueling_deep_q_network_dense_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ü
_input_shapesê
ç: :	::
::	::	:: : : : : : : :	::
::	::	::	::
::	::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	: 

_output_shapes
:: 

_output_shapes
: 
Þ	
Ë
4__inference_dueling_deep_q_network_layer_call_fn_756
input_1
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_dueling_deep_q_network_layer_call_and_return_conditional_losses_743`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1


ñ
>__inference_dense_layer_call_and_return_conditional_losses_696

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø	
É
4__inference_dueling_deep_q_network_layer_call_fn_769	
state
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_dueling_deep_q_network_layer_call_and_return_conditional_losses_743`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namestate


O__inference_dueling_deep_q_network_layer_call_and_return_conditional_losses_743	
state"
dense_407973860:	
dense_407973862:	%
dense_1_407973877:
 
dense_1_407973879:	$
dense_2_407973893:	
dense_2_407973895:$
dense_3_407973909:	
dense_3_407973911:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCallç
dense/StatefulPartitionedCallStatefulPartitionedCallstatedense_407973860dense_407973862*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_696
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_407973877dense_1_407973879*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_382
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_407973893dense_2_407973895*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_399
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_3_407973909dense_3_407973911*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_3_layer_call_and_return_conditional_losses_678X
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
MeanMean(dense_3/StatefulPartitionedCall:output:0Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(u
subSub(dense_3/StatefulPartitionedCall:output:0Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
addAddV2(dense_2/StatefulPartitionedCall:output:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namestate
£

ô
@__inference_dense_1_layer_call_and_return_conditional_losses_382

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡

O__inference_dueling_deep_q_network_layer_call_and_return_conditional_losses_723
input_1"
dense_407974002:	
dense_407974004:	%
dense_1_407974007:
 
dense_1_407974009:	$
dense_2_407974012:	
dense_2_407974014:$
dense_3_407974017:	
dense_3_407974019:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCallé
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_407974002dense_407974004*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_696
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_407974007dense_1_407974009*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_382
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_407974012dense_2_407974014*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_399
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_3_407974017dense_3_407974019*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_3_layer_call_and_return_conditional_losses_678X
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
MeanMean(dense_3/StatefulPartitionedCall:output:0Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(u
subSub(dense_3/StatefulPartitionedCall:output:0Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
addAddV2(dense_2/StatefulPartitionedCall:output:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
°%
´
O__inference_dueling_deep_q_network_layer_call_and_return_conditional_losses_535	
state7
$dense_matmul_readvariableop_resource:	4
%dense_biasadd_readvariableop_resource:	:
&dense_1_matmul_readvariableop_resource:
6
'dense_1_biasadd_readvariableop_resource:	9
&dense_2_matmul_readvariableop_resource:	5
'dense_2_biasadd_readvariableop_resource:9
&dense_3_matmul_readvariableop_resource:	5
'dense_3_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0u
dense/MatMulMatMulstate#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_3/MatMulMatMuldense_1/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
MeanMeandense_3/BiasAdd:output:0Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(e
subSubdense_3/BiasAdd:output:0Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
addAddV2dense_2/BiasAdd:output:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namestate"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*©
serving_default
9
args_0/
serving_default_args_0:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ÊU
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

dense1

	dense2

V
A
	optimizer

signatures
#_self_saveable_object_factories"
_tf_keras_model
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
½
trace_0
trace_12
4__inference_dueling_deep_q_network_layer_call_fn_756
4__inference_dueling_deep_q_network_layer_call_fn_769
²
FullArgSpec
args	
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
ó
trace_0
trace_12¼
O__inference_dueling_deep_q_network_layer_call_and_return_conditional_losses_535
O__inference_dueling_deep_q_network_layer_call_and_return_conditional_losses_723
²
FullArgSpec
args	
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
ÏBÌ
%__inference__wrapped_model_1811993765args_0"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
à
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

kernel
bias
#&_self_saveable_object_factories"
_tf_keras_layer
à
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

kernel
bias
#-_self_saveable_object_factories"
_tf_keras_layer
à
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

kernel
bias
#4_self_saveable_object_factories"
_tf_keras_layer
à
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

kernel
bias
#;_self_saveable_object_factories"
_tf_keras_layer
ã
<iter

=beta_1

>beta_2
	?decay
@learning_ratem[m\m]m^m_m`mambvcvdvevfvgvhvivj"
	optimizer
,
Aserving_default"
signature_map
 "
trackable_dict_wrapper
6:4	2#dueling_deep_q_network/dense/kernel
0:.2!dueling_deep_q_network/dense/bias
9:7
2%dueling_deep_q_network/dense_1/kernel
2:02#dueling_deep_q_network/dense_1/bias
8:6	2%dueling_deep_q_network/dense_2/kernel
1:/2#dueling_deep_q_network/dense_2/bias
8:6	2%dueling_deep_q_network/dense_3/kernel
1:/2#dueling_deep_q_network/dense_3/bias
 "
trackable_list_wrapper
<
0
	1

2
3"
trackable_list_wrapper
'
B0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
4__inference_dueling_deep_q_network_layer_call_fn_756input_1"
²
FullArgSpec
args	
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÜBÙ
4__inference_dueling_deep_q_network_layer_call_fn_769state"
²
FullArgSpec
args	
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
O__inference_dueling_deep_q_network_layer_call_and_return_conditional_losses_535state"
²
FullArgSpec
args	
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
O__inference_dueling_deep_q_network_layer_call_and_return_conditional_losses_723input_1"
²
FullArgSpec
args	
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_dict_wrapper
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÎBË
(__inference_signature_wrapper_1811993794args_0"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
N
W	variables
X	keras_api
	Ytotal
	Zcount"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
Y0
Z1"
trackable_list_wrapper
-
W	variables"
_generic_user_object
:  (2total
:  (2count
;:9	2*Adam/dueling_deep_q_network/dense/kernel/m
5:32(Adam/dueling_deep_q_network/dense/bias/m
>:<
2,Adam/dueling_deep_q_network/dense_1/kernel/m
7:52*Adam/dueling_deep_q_network/dense_1/bias/m
=:;	2,Adam/dueling_deep_q_network/dense_2/kernel/m
6:42*Adam/dueling_deep_q_network/dense_2/bias/m
=:;	2,Adam/dueling_deep_q_network/dense_3/kernel/m
6:42*Adam/dueling_deep_q_network/dense_3/bias/m
;:9	2*Adam/dueling_deep_q_network/dense/kernel/v
5:32(Adam/dueling_deep_q_network/dense/bias/v
>:<
2,Adam/dueling_deep_q_network/dense_1/kernel/v
7:52*Adam/dueling_deep_q_network/dense_1/bias/v
=:;	2,Adam/dueling_deep_q_network/dense_2/kernel/v
6:42*Adam/dueling_deep_q_network/dense_2/bias/v
=:;	2,Adam/dueling_deep_q_network/dense_3/kernel/v
6:42*Adam/dueling_deep_q_network/dense_3/bias/v
%__inference__wrapped_model_1811993765p/¢,
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ´
O__inference_dueling_deep_q_network_layer_call_and_return_conditional_losses_535a.¢+
$¢!

stateÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¶
O__inference_dueling_deep_q_network_layer_call_and_return_conditional_losses_723c0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
4__inference_dueling_deep_q_network_layer_call_fn_756V0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
4__inference_dueling_deep_q_network_layer_call_fn_769T.¢+
$¢!

stateÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
(__inference_signature_wrapper_1811993794z9¢6
¢ 
/ª,
*
args_0 
args_0ÿÿÿÿÿÿÿÿÿ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ