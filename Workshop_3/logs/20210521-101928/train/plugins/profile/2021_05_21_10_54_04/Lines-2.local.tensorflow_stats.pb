"?f
BHostIDLE"IDLE1    ???@A    ???@a=?????i=??????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ؍@9     ؍@A     ؍@I     ؍@a?}?$??iޘ?Cc???Unknown?
iHostWriteSummary"WriteSummary(1      D@9      D@A      D@I      D@a?=i?}?n?i?=????Unknown?
yHost_FusedMatMul"sequential_33/dense_103/BiasAdd(1      =@9      =@A      =@I      =@a??o?tf?i??鱘???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      ?@9      ?@A      <@I      <@a?w????e?iGk??`????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      3@9      3@A      3@I      3@af??m]?i???????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      3@9      3@A      3@I      3@af??m]?i\̑?????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      2@9      2@A      2@I      2@aQx???[?i8?׽????Unknown
g	HostStridedSlice"strided_slice(1      2@9      2@A      2@I      2@aQx???[?ia?h?????Unknown
V
HostMean"Mean(1      .@9      .@A      .@I      .@aC?-;W?i?[?K????Unknown
eHost
LogicalAnd"
LogicalAnd(1      ,@9      ,@A      ,@I      ,@a?w????U?i???"????Unknown?
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      $@9      $@A      $@I      $@a?=i?}?N?ic=]????Unknown
?HostMatMul".gradient_tape/sequential_33/dense_104/MatMul_1(1      $@9      $@A      $@I      $@a?=i?}?N?i?[??????Unknown
?HostMatMul",gradient_tape/sequential_33/dense_105/MatMul(1      $@9      $@A      $@I      $@a?=i?}?N?i?^???Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@aQx???K?iԬ>V???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@aQx???K?i)?SaN#???Unknown
?HostMatMul",gradient_tape/sequential_33/dense_104/MatMul(1      "@9      "@A      "@I      "@aQx???K?i=??F*???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @a?d????H?i??ix0???Unknown
?HostMatMul",gradient_tape/sequential_33/dense_103/MatMul(1       @9       @A       @I       @a?d????H?i???O?6???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a?w????E?i?y??<???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a?w????E?i+(??A???Unknown
dHostDataset"Iterator::Model(1      9@9      9@A      @I      @a?w????E?i??TK?F???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a?w????E?igj??XL???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?w????E?i???Q???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a?w????E?i???F0W???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?w????E?iA[??\???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @ai?????B?i??v\Aa???Unknown
?HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @ai?????B?i.???e???Unknown
?HostBiasAddGrad"9gradient_tape/sequential_33/dense_105/BiasAdd/BiasAddGrad(1      @9      @A      @I      @ai?????B?ij?U5?j???Unknown
yHost_FusedMatMul"sequential_33/dense_104/BiasAdd(1      @9      @A      @I      @ai?????B?i? š1o???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?=i?}?>?i?-w?s???Unknown
? HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?=i?}?>?i[)?v???Unknown
?!HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a?=i?}?>?iE??0?z???Unknown
V"HostSum"Sum_2(1      @9      @A      @I      @a?=i?}?>?im??`?~???Unknown
v#HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?=i?}?>?i?????????Unknown
?$HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?=i?}?>?i???l????Unknown
?%HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?=i?}?>?i?<??K????Unknown
?&HostBiasAddGrad"9gradient_tape/sequential_33/dense_104/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?=i?}?>?ijV+????Unknown
?'HostReadVariableOp"-sequential_33/dense_103/MatMul/ReadVariableOp(1      @9      @A      @I      @a?=i?}?>?i5?O
????Unknown
t(HostSigmoid"sequential_33/dense_105/Sigmoid(1      @9      @A      @I      @a?=i?}?>?i]ĺ~?????Unknown
\)HostGreater"Greater(1      @9      @A      @I      @a?d????8?iJ??q????Unknown
v*HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a?d????8?i7??d????Unknown
v+HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?d????8?i$??W4????Unknown
?,HostBiasAddGrad"9gradient_tape/sequential_33/dense_103/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?d????8?i??JM????Unknown
y-Host_FusedMatMul"sequential_33/dense_105/BiasAdd(1      @9      @A      @I      @a?d????8?i?x?=f????Unknown
v.HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @ai?????2?i?-???????Unknown
X/HostCast"Cast_3(1      @9      @A      @I      @ai?????2?i`???????Unknown
X0HostEqual"Equal(1      @9      @A      @I      @ai?????2?i?*`^????Unknown
?1HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      ??A      @I      ??ai?????2?i?Kb?????Unknown
s2HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @ai?????2?is ??????Unknown
u3HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @ai?????2?i$?тV????Unknown
d4HostAddN"SGD/gradients/AddN(1      @9      @A      @I      @ai?????2?i?i	9?????Unknown
j5HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @ai?????2?i?A??????Unknown
r6HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @ai?????2?i7?x?N????Unknown
z7HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @ai?????2?i臰[?????Unknown
~8HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @ai?????2?i?<??????Unknown
~9HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @ai?????2?iJ??F????Unknown
?:Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @ai?????2?i??W~?????Unknown
?;Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @ai?????2?i?Z?4?????Unknown
?<HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @ai?????2?i]??>????Unknown
?=HostMatMul".gradient_tape/sequential_33/dense_105/MatMul_1(1      @9      @A      @I      @ai?????2?i????????Unknown
n>HostTanh"sequential_33/dense_103/Tanh(1      @9      @A      @I      @ai?????2?i?x6W?????Unknown
??HostReadVariableOp".sequential_33/dense_105/BiasAdd/ReadVariableOp(1      @9      @A      @I      @ai?????2?ip-n7????Unknown
t@HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a?d????(?i?????????Unknown
vAHostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a?d????(?i\c P????Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a?d????(?iҖ?y?????Unknown
VCHostCast"Cast(1       @9       @A       @I       @a?d????(?iHX?h????Unknown
XDHostCast"Cast_4(1       @9       @A       @I       @a?d????(?i???l?????Unknown
XEHostCast"Cast_5(1       @9       @A       @I       @a?d????(?i4 M??????Unknown
vFHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?d????(?i?x?_????Unknown
vGHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a?d????(?i ?Aٚ????Unknown
?HHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?d????(?i?i?R'????Unknown
`IHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?d????(?i?6̳????Unknown
?JHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a?d????(?i?Z?E@????Unknown
xKHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?d????(?i??+??????Unknown
?LHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a?d????(?inK?8Y????Unknown
?MHostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a?d????(?i?? ??????Unknown
?NHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a?d????(?iZ<?+r????Unknown
?OHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?d????(?iд??????Unknown
?PHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a?d????(?iF-??????Unknown
?QHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a?d????(?i??
?????Unknown
?RHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a?d????(?i2??????Unknown
~SHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?d????(?i????0????Unknown
?THostTanhGrad".gradient_tape/sequential_33/dense_103/TanhGrad(1       @9       @A       @I       @a?d????(?iz?????Unknown
?UHostTanhGrad".gradient_tape/sequential_33/dense_104/TanhGrad(1       @9       @A       @I       @a?d????(?i???}I????Unknown
?VHostReadVariableOp".sequential_33/dense_103/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?d????(?i
 o??????Unknown
?WHostReadVariableOp".sequential_33/dense_104/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?d????(?i?x?pb????Unknown
?XHostReadVariableOp"-sequential_33/dense_105/MatMul/ReadVariableOp(1       @9       @A       @I       @a?d????(?i??c??????Unknown
aYHostIdentity"Identity(1      ??9      ??A      ??I      ??a?d?????i1-!'?????Unknown?
TZHostMul"Mul(1      ??9      ??A      ??I      ??a?d?????ili?c{????Unknown
|[HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a?d?????i????A????Unknown
}\HostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?d?????i??X?????Unknown
u]HostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a?d?????i?????Unknown
b^HostDivNoNan"div_no_nan_1(1      ??9      ??A      ??I      ??a?d?????iXZ?V?????Unknown
w_HostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??a?d?????i????Z????Unknown
?`HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?d?????i??M? ????Unknown
?aHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a?d?????i	?????Unknown
?bHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?d?????iDK?I?????Unknown
?cHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?d?????i???s????Unknown
?dHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a?d?????i??B?9????Unknown
?eHostReadVariableOp"-sequential_33/dense_104/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?d?????i?????????Unknown
nfHostTanh"sequential_33/dense_104/Tanh(1      ??9      ??A      ??I      ??a?d?????i?^c ???Unknown
JgHostReadVariableOp"div_no_nan/ReadVariableOp_1(i?^c ???Unknown
LhHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i?^c ???Unknown
[iHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i?^c ???Unknown
[jHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i?^c ???Unknown*?f
uHostFlushSummaryWriter"FlushSummaryWriter(1     ؍@9     ؍@A     ؍@I     ؍@a_?z???i_?z????Unknown?
iHostWriteSummary"WriteSummary(1      D@9      D@A      D@I      D@a?[?*&???i<di?+b???Unknown?
yHost_FusedMatMul"sequential_33/dense_103/BiasAdd(1      =@9      =@A      =@I      =@a=o)ߎ???i??b"????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      ?@9      ?@A      <@I      <@a?Y?A??i?bS+ě???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      3@9      3@A      3@I      3@a?0ը
"??iG??UL???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      3@9      3@A      3@I      3@a?0ը
"??i
???l???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      2@9      2@A      2@I      2@a??&????i!?4=?????Unknown
gHostStridedSlice"strided_slice(1      2@9      2@A      2@I      2@a??&????i8????2???Unknown
V	HostMean"Mean(1      .@9      .@A      .@I      .@a??<?????iKvPlj????Unknown
e
Host
LogicalAnd"
LogicalAnd(1      ,@9      ,@A      ,@I      ,@a?Y?A??i???pp????Unknown?
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      $@9      $@A      $@I      $@a?[?*&?{?ii??t	???Unknown
?HostMatMul".gradient_tape/sequential_33/dense_104/MatMul_1(1      $@9      $@A      $@I      $@a?[?*&?{?i ?t	y@???Unknown
?HostMatMul",gradient_tape/sequential_33/dense_105/MatMul(1      $@9      $@A      $@I      $@a?[?*&?{?i׳?U}w???Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a??&??x?i?4????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a??&??x?i?oe?????Unknown
?HostMatMul",gradient_tape/sequential_33/dense_104/MatMul(1      "@9      "@A      "@I      "@a??&??x?i?Ͳ????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @a??b"?v?iW??`8???Unknown
?HostMatMul",gradient_tape/sequential_33/dense_103/MatMul(1       @9       @A       @I       @a??b"?v?i?X<?d???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a?Y?As?ii?xӒ????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a?Y?As?i???????Unknown
dHostDataset"Iterator::Model(1      9@9      9@A      @I      @a?Y?As?i???ט????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a?Y?As?i?-?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?Y?As?i58iܞ$???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a?Y?As?i?d??!K???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?Y?As?i?????q???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a??J?p?i?%u?????Unknown
?HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a??J?p?i??H	?????Unknown
?HostBiasAddGrad"9gradient_tape/sequential_33/dense_105/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??J?p?i?M|??????Unknown
yHost_FusedMatMul"sequential_33/dense_104/BiasAdd(1      @9      @A      @I      @a??J?p?i???1?????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?[?*&?k?i??W1???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?[?*&?k?is?~?,???Unknown
? HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a?[?*&?k?i??0?5H???Unknown
V!HostSum"Sum_2(1      @9      @A      @I      @a?[?*&?k?i+?[ʷc???Unknown
v"HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?[?*&?k?i?ʆ?9???Unknown
?#HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?[?*&?k?i?ű?????Unknown
?$HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?[?*&?k?i???<>????Unknown
?%HostBiasAddGrad"9gradient_tape/sequential_33/dense_104/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?[?*&?k?i??c?????Unknown
?&HostReadVariableOp"-sequential_33/dense_103/MatMul/ReadVariableOp(1      @9      @A      @I      @a?[?*&?k?i??2?B????Unknown
t'HostSigmoid"sequential_33/dense_105/Sigmoid(1      @9      @A      @I      @a?[?*&?k?iS?]?????Unknown
\(HostGreater"Greater(1      @9      @A      @I      @a??b"?f?i?g????Unknown
v)HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a??b"?f?i?x??4???Unknown
v*HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a??b"?f?ic????J???Unknown
?+HostBiasAddGrad"9gradient_tape/sequential_33/dense_103/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??b"?f?i>???`???Unknown
y,Host_FusedMatMul"sequential_33/dense_105/BiasAdd(1      @9      @A      @I      @a??b"?f?ià	H?v???Unknown
v-HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a??J?`?i?j#?N????Unknown
X.HostCast"Cast_3(1      @9      @A      @I      @a??J?`?i?4=?ϗ???Unknown
X/HostEqual"Equal(1      @9      @A      @I      @a??J?`?i??V&Q????Unknown
?0HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      ??A      @I      ??a??J?`?i??ppҸ???Unknown
s1HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a??J?`?iג??S????Unknown
u2HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a??J?`?i?\??????Unknown
d3HostAddN"SGD/gradients/AddN(1      @9      @A      @I      @a??J?`?i?&?NV????Unknown
j4HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a??J?`?i??ט?????Unknown
r5HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a??J?`?i????X???Unknown
z6HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a??J?`?i??-????Unknown
~7HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a??J?`?i?N%w[,???Unknown
~8HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a??J?`?i????<???Unknown
?9Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a??J?`?i??X^M???Unknown
?:Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a??J?`?i??rU?]???Unknown
?;HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a??J?`?i?v??`n???Unknown
?<HostMatMul".gradient_tape/sequential_33/dense_105/MatMul_1(1      @9      @A      @I      @a??J?`?iA???~???Unknown
n=HostTanh"sequential_33/dense_103/Tanh(1      @9      @A      @I      @a??J?`?i?3c????Unknown
?>HostReadVariableOp".sequential_33/dense_105/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??J?`?i??}?????Unknown
t?HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a??b"?V?ic?Y?????Unknown
v@HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a??b"?V?i?7?5?????Unknown
vAHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a??b"?V?ii?????Unknown
VBHostCast"Cast(1       @9       @A       @I       @a??b"?V?ik???????Unknown
XCHostCast"Cast_4(1       @9       @A       @I       @a??b"?V?i??/??????Unknown
XDHostCast"Cast_5(1       @9       @A       @I       @a??b"?V?i?@??????Unknown
vEHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a??b"?V?is.R??????Unknown
vFHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a??b"?V?i?_c^?????Unknown
?GHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a??b"?V?i#?t:????Unknown
`HHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a??b"?V?i{????Unknown
?IHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a??b"?V?i????????Unknown
xJHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a??b"?V?i+%???#???Unknown
?KHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a??b"?V?i?V???.???Unknown
?LHostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a??b"?V?iۇʆ?9???Unknown
?MHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a??b"?V?i3??b?D???Unknown
?NHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a??b"?V?i???>?O???Unknown
?OHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a??b"?V?i???Z???Unknown
?PHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a??b"?V?i;M??e???Unknown
?QHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a??b"?V?i?~ ??p???Unknown
~RHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a??b"?V?i??1??{???Unknown
?SHostTanhGrad".gradient_tape/sequential_33/dense_103/TanhGrad(1       @9       @A       @I       @a??b"?V?iC?B??????Unknown
?THostTanhGrad".gradient_tape/sequential_33/dense_104/TanhGrad(1       @9       @A       @I       @a??b"?V?i?Tg?????Unknown
?UHostReadVariableOp".sequential_33/dense_103/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a??b"?V?i?CeC?????Unknown
?VHostReadVariableOp".sequential_33/dense_104/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a??b"?V?iKuv?????Unknown
?WHostReadVariableOp"-sequential_33/dense_105/MatMul/ReadVariableOp(1       @9       @A       @I       @a??b"?V?i?????????Unknown
aXHostIdentity"Identity(1      ??9      ??A      ??I      ??a??b"?F?iO??iz????Unknown?
TYHostMul"Mul(1      ??9      ??A      ??I      ??a??b"?F?i?ט??????Unknown
|ZHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a??b"?F?i?p?E{????Unknown
}[HostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a??b"?F?iS	???????Unknown
u\HostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a??b"?F?i???!|????Unknown
b]HostDivNoNan"div_no_nan_1(1      ??9      ??A      ??I      ??a??b"?F?i?:???????Unknown
w^HostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??a??b"?F?iW???|????Unknown
?_HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a??b"?F?il?k?????Unknown
?`Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a??b"?F?i???}????Unknown
?aHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a??b"?F?i[??G?????Unknown
?bHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a??b"?F?i6??~????Unknown
?cHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a??b"?F?i???#?????Unknown
?dHostReadVariableOp"-sequential_33/dense_104/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a??b"?F?i_g??????Unknown
neHostTanh"sequential_33/dense_104/Tanh(1      ??9      ??A      ??I      ??a??b"?F?i     ???Unknown
JfHostReadVariableOp"div_no_nan/ReadVariableOp_1(i     ???Unknown
LgHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i     ???Unknown
[hHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i     ???Unknown
[iHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i     ???Unknown2CPU