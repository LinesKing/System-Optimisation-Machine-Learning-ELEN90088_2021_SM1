"?b
BHostIDLE"IDLE1     ??@A     ??@a??,}??i??,}???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?T@9     ?T@A     ?T@I     ?T@a7 ?@i??i?T
v ???Unknown
dHostDataset"Iterator::Model(1     ?P@9     ?P@A     ?P@I     ?P@aԂ,??y??i?~-?E????Unknown
oHost_FusedMatMul"sequential/dense/Relu(1     ?D@9     ?D@A     ?D@I     ?D@a?? <L*??i???????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?@@9     ?@@A     ?@@I     ?@@a?G*;??i????5???Unknown?
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      1@9      1@A      1@I      1@a?K??p?i/O??LW???Unknown
tHostAssignAddVariableOp"AssignAddVariableOp(1      (@9      (@A      (@I      (@a?9?k??g?iiJ??n???Unknown
iHostWriteSummary"WriteSummary(1      $@9      $@A      $@I      $@aJ??Yc?c?i?ǣ?????Unknown?
t	Host_FusedMatMul"sequential/dense_2/BiasAdd(1      $@9      $@A      $@I      $@aJ??Yc?c?is??iA????Unknown
g
HostStridedSlice"strided_slice(1      $@9      $@A      $@I      $@aJ??Yc?c?i?LW??????Unknown
`HostGatherV2"
GatherV2_1(1       @9       @A       @I       @a???8z_?i??骹???Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a???8z_?i???h????Unknown
?HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a???8z_?ik?."%????Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1       @9       @A       @I       @a???8z_?i<?v>?????Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a???}??[?i3]5??????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a???}??[?i*2?/m???Unknown
VHostMean"Mean(1      @9      @A      @I      @a???}??[?i!??2???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a???}??[?i?q!????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a???}??[?i?0??-???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @a???}??[?i???;???Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a???}??[?i?Z??HI???Unknown
aHostCast"sequential/Cast(1      @9      @A      @I      @a???}??[?i?/mW???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?9?k??W?i???b???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a?9?k??W?i.?خ?n???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a?9?k??W?iK??wz???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?9?k??W?ih?DYE????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @a?9?k??W?i?wz.????Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      @9      @A      @I      @a?9?k??W?i?R??????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @aJ??Yc?S?i?3]5?????Unknown
VHostSum"Sum_2(1      @9      @A      @I      @aJ??Yc?S?i(
g?????Unknown
jHostMean"binary_crossentropy/Mean(1      @9      @A      @I      @aJ??Yc?S?ik???c????Unknown
? HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @aJ??Yc?S?i??c?9????Unknown
?!Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @aJ??Yc?S?i???????Unknown
y"HostMatMul"%gradient_tape/sequential/dense/MatMul(1      @9      @A      @I      @aJ??Yc?S?i4??-?????Unknown
o#HostSigmoid"sequential/dense_2/Sigmoid(1      @9      @A      @I      @aJ??Yc?S?iw{j_?????Unknown
e$Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a???8zO?i?b???????Unknown?
z%HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a???8zO?iIJ?{y????Unknown
v&HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a???8zO?i?1?	X????Unknown
v'HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a???8zO?i??6???Unknown
v(HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a???8zO?i? &
???Unknown
?)HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a???8zO?i??A?????Unknown
?*HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a???8zO?iV?eB????Unknown
?+HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???8zO?i???а!???Unknown
?,HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???8zO?i(??^?)???Unknown
v-HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?9?k??G?i??HIv/???Unknown
X.HostCast"Cast_4(1      @9      @A      @I      @a?9?k??G?iDy?3]5???Unknown
?/HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      ??A      @I      ??a?9?k??G?i?f~D;???Unknown
u0HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?9?k??G?i`T	+A???Unknown
?1HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?9?k??G?i?A??G???Unknown
?2HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a?9?k??G?i|/O??L???Unknown
~3HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?9?k??G?i
???R???Unknown
v4HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a?9?k??G?i?
???X???Unknown
`5HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a?9?k??G?i&???^???Unknown
?6HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a?9?k??G?i?庈?d???Unknown
?7Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a?9?k??G?iB?Us{j???Unknown
~8HostRealDiv")gradient_tape/binary_crossentropy/truediv(1      @9      @A      @I      @a?9?k??G?i???]bp???Unknown
?9HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?9?k??G?i^??HIv???Unknown
?:HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?9?k??G?i??&30|???Unknown
?;HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?9?k??G?iz??????Unknown
v<HostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a???8z??i.}?d????Unknown
V=HostCast"Cast(1       @9       @A       @I       @a???8z??i?p???????Unknown
X>HostCast"Cast_2(1       @9       @A       @I       @a???8z??i?d???????Unknown
X?HostEqual"Equal(1       @9       @A       @I       @a???8z??iJX	:ԑ???Unknown
\@HostGreater"Greater(1       @9       @A       @I       @a???8z??i?K?Õ???Unknown
sAHostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a???8z??i??-Ȳ????Unknown
dBHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a???8z??if3??????Unknown
rCHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a???8z??i'QV?????Unknown
?DHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a???8z??i?c??????Unknown
}EHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a???8z??i?u?o????Unknown
uFHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a???8z??i6?+_????Unknown
bGHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a???8z??i???rN????Unknown
wHHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a???8z??i?骹=????Unknown
xIHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a???8z??iRݼ -????Unknown
~JHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a???8z??i??G????Unknown
?KHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a???8z??i????????Unknown
?LHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a???8z??in????????Unknown
?MHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a???8z??i"??????Unknown
?NHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a???8z??i֟d?????Unknown
}OHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @a???8z??i??(??????Unknown
PHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @a???8z??i>?:??????Unknown
?QHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1       @9       @A       @I       @a???8z??i?zL9?????Unknown
?RHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a???8z??i?n^??????Unknown
?SHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1       @9       @A       @I       @a???8z??iZbpǅ????Unknown
vTHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a???8z/?i4\?j}????Unknown
vUHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a???8z/?iV?u????Unknown
XVHostCast"Cast_3(1      ??9      ??A      ??I      ??a???8z/?i?O?l????Unknown
aWHostIdentity"Identity(1      ??9      ??A      ??I      ??a???8z/?i?I?Ud????Unknown?
TXHostMul"Mul(1      ??9      ??A      ??I      ??a???8z/?i?C?[????Unknown
|YHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a???8z/?iv=??S????Unknown
vZHostExp"%binary_crossentropy/logistic_loss/Exp(1      ??9      ??A      ??I      ??a???8z/?iP7/@K????Unknown
w[HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a???8z/?i*1??B????Unknown
y\HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a???8z/?i+A?:????Unknown
?]HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a???8z/?i?$?*2????Unknown
?^Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a???8z/?i?S?)????Unknown
?_HostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??a???8z/?i??q!????Unknown
?`HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a???8z/?ile????Unknown
?aHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a???8z/?iF??????Unknown
?bHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a???8z/?i w\????Unknown
?cHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a???8z/?i?????????Unknown
WdHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i?????????Unknown
[eHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i?????????Unknown
YfHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown*?a
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?T@9     ?T@A     ?T@I     ?T@a?βrK??i?βrK???Unknown
dHostDataset"Iterator::Model(1     ?P@9     ?P@A     ?P@I     ?P@a??#JB8??iX?K?Z???Unknown
oHost_FusedMatMul"sequential/dense/Relu(1     ?D@9     ?D@A     ?D@I     ?D@a????x???i?X????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?@@9     ?@@A     ?@@I     ?@@aJ?o ?Ȭ?i?|f(l???Unknown?
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      1@9      1@A      1@I      1@a??ד秝?i?????F???Unknown
tHostAssignAddVariableOp"AssignAddVariableOp(1      (@9      (@A      (@I      (@a{???????i?S?u?????Unknown
iHostWriteSummary"WriteSummary(1      $@9      $@A      $@I      $@a<?$??q??i????????Unknown?
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1      $@9      $@A      $@I      $@a<?$??q??i?D??????Unknown
g	HostStridedSlice"strided_slice(1      $@9      $@A      $@I      $@a<?$??q??i(2??????Unknown
`
HostGatherV2"
GatherV2_1(1       @9       @A       @I       @a?imR???ixm??8????Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a?imR???idԫ??L???Unknown
?HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a?imR???i?_?f????Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1       @9       @A       @I       @a?imR???i?6,???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a?|f(l??i???ּ????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a?|f(l??i?Cwm????Unknown
VHostMean"Mean(1      @9      @A      @I      @a?|f(l??i?݌Q???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a?|f(l??i?w
?β???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?|f(l??is?X???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @a?|f(l??if??/v???Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a?|f(l??iYE???????Unknown
aHostCast"sequential/Cast(1      @9      @A      @I      @a?|f(l??iL? :?9???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a{???????i??G1M????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a{???????i??(	????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a{???????i"??4???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a{???????iD8?????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @a{???????i?Nd=????Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      @9      @A      @I      @a{???????i?d??/???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a<?$??q??iI??S?u???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a<?$??q??i҉̡?????Unknown
jHostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a<?$??q??i[??N???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a<?$??q??i???=G???Unknown
? Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a<?$??q??imA??݌???Unknown
y!HostMatMul"%gradient_tape/sequential/dense/MatMul(1      @9      @A      @I      @a<?$??q??i??ڤ????Unknown
o"HostSigmoid"sequential/dense_2/Sigmoid(1      @9      @A      @I      @a<?$??q??if(l???Unknown
e#Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?imR?{?iSu??>P???Unknown?
z$HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a?imR?{?i'??q????Unknown
v%HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a?imR?{?i????????Unknown
v&HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?imR?{?iϡ???????Unknown
v'HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?imR?{?i??a`?/???Unknown
?(HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?imR?{?iw?;\g???Unknown
?)HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?imR?{?iK??.????Unknown
?*HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?imR?{?i??N????Unknown
?+HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?imR?{?i????????Unknown
v,HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a{?????t?iwm??8???Unknown
X-HostCast"Cast_4(1      @9      @A      @I      @a{?????t?i1??b???Unknown
?.HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      ??A      @I      ??a{?????t?iP???m????Unknown
u/HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a{?????t?ioX?K????Unknown
?0HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a{?????t?i????)????Unknown
?1HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a{?????t?i?.??
???Unknown
~2HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a{?????t?i̹B??3???Unknown
v3HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a{?????t?i?D???]???Unknown
`4HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a{?????t?i
Љ̡????Unknown
?5HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a{?????t?i)[-?????Unknown
?6Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a{?????t?iH???]????Unknown
~7HostRealDiv")gradient_tape/binary_crossentropy/truediv(1      @9      @A      @I      @a{?????t?igqt?;???Unknown
?8HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a{?????t?i???/???Unknown
?9HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a{?????t?i?????X???Unknown
?:HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a{?????t?i?_?Ղ???Unknown
v;HostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a?imR?k?i.??????Unknown
V<HostCast"Cast(1       @9       @A       @I       @a?imR?k?i?!9W?????Unknown
X=HostCast"Cast_2(1       @9       @A       @I       @a?imR?k?i)???????Unknown
X>HostEqual"Equal(1       @9       @A       @I       @a?imR?k?il0?z????Unknown
\?HostGreater"Greater(1       @9       @A       @I       @a?imR?k?i?7?Nd???Unknown
s@HostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a?imR?k?i@???M*???Unknown
dAHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a?imR?k?i?FZ?6F???Unknown
rBHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a?imR?k?iN?E b???Unknown
?CHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?imR?k?i~U4?	~???Unknown
}DHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a?imR?k?i?\???????Unknown
uEHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a?imR?k?iRd=ܵ???Unknown
bFHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a?imR?k?i?k{??????Unknown
wGHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?imR?k?i&s???????Unknown
xHHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?imR?k?i?zU4?	???Unknown
~IHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a?imR?k?i???%???Unknown
?JHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a?imR?k?id?/?jA???Unknown
?KHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a?imR?k?iΐ?+T]???Unknown
?LHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a?imR?k?i8?	~=y???Unknown
?MHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?imR?k?i??v?&????Unknown
}NHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @a?imR?k?i??"????Unknown
OHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @a?imR?k?iv?Pu?????Unknown
?PHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1       @9       @A       @I       @a?imR?k?iൽ??????Unknown
?QHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?imR?k?iJ?*????Unknown
?RHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1       @9       @A       @I       @a?imR?k?i?ėl? ???Unknown
vSHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?imR?[?iiH??.???Unknown
vTHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a?imR?[?i???<???Unknown
XUHostCast"Cast_3(1      ??9      ??A      ??I      ??a?imR?[?i?O;h?J???Unknown
aVHostIdentity"Identity(1      ??9      ??A      ??I      ??a?imR?[?i??q?X???Unknown?
TWHostMul"Mul(1      ??9      ??A      ??I      ??a?imR?[?i=W??|f???Unknown
|XHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a?imR?[?i???cqt???Unknown
vYHostExp"%binary_crossentropy/logistic_loss/Exp(1      ??9      ??A      ??I      ??a?imR?[?i?^f????Unknown
wZHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?imR?[?i\?K?Z????Unknown
y[HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?imR?[?if?_O????Unknown
?\HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a?imR?[?i???D????Unknown
?]Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a?imR?[?i{m??8????Unknown
?^HostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??a?imR?[?i0?%[-????Unknown
?_HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?imR?[?i?t\"????Unknown
?`HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?imR?[?i????????Unknown
?aHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a?imR?[?iO|?V????Unknown
?bHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?imR?[?i     ???Unknown
WcHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i     ???Unknown
[dHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i     ???Unknown
YeHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i     ???Unknown2CPU