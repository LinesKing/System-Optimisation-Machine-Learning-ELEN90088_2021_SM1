"?b
BHostIDLE"IDLE1     ¬@A     ¬@apa\s9o??ipa\s9o???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     ?b@9     ?b@A     ?b@I     ?b@a9o?`??id???O???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     @_@9     @_@A     @_@I     @_@a?׷0????i ???e???Unknown
dHostDataset"Iterator::Model(1     ?h@9     ?h@A      I@I      I@a???????ikӄ?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?@@9     ?@@A     ?@@I     ?@@a?|U~?iK񟴳????Unknown?
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1     ?@@9     ?@@A     ?@@I     ?@@a?|U~?i+??^:???Unknown
VHostMean"Mean(1      6@9      6@A      6@I      6@a񟴳?8t?ikx"??b???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      .@9      .@A      .@I      .@a}????k?i?(?1d~???Unknown
q	Host_FusedMatMul"sequential/dense_1/Relu(1      ,@9      ,@A      ,@I      ,@a??q???i?in?p!????Unknown
V
HostCast"Cast(1      &@9      &@A      &@I      &@a񟴳?8d?iO$Z????Unknown
iHostWriteSummary"WriteSummary(1      &@9      &@A      &@I      &@a񟴳?8d?i???????Unknown?
?HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      &@9      &@A      &@I      &@a񟴳?8d?iN???????Unknown
`HostDivNoNan"
div_no_nan(1      &@9      &@A      &@I      &@a񟴳?8d?i?l?????Unknown
?Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      &@9      &@A      &@I      &@a񟴳?8d?i?!?>????Unknown
?HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      &@9      &@A      &@I      &@a񟴳?8d?i.֦	w???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @aHt?3+j]?i??@, ???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @aHt?3+j]?i???4?.???Unknown
oHost_FusedMatMul"sequential/dense/Relu(1       @9       @A       @I       @aHt?3+j]?i\?tJ?=???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a??q???Y?iv[?tJ???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a??q???Y?i?.B0SW???Unknown
vHostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a??q???Y?i??(?1d???Unknown
?HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      @9      @A      @I      @a??q???Y?ih?q???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @a??q???Y?i+Y???}???Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a??q???Y?i???̊???Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @a??q???Y?i???n?????Unknown
lHostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a6??f?V?i}D?>?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a6??f?V?iI?*?????Unknown
jHostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a6??f?V?i8^?¸???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @a6??f?V?i᱑??????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a6??f?V?i?+??????Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      @9      @A      @I      @a6??f?V?iy??O?????Unknown
? HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a??u [bR?iM?x}????Unknown
?!HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a??u [bR?i!??<????Unknown
?"HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??u [bR?i?Uy?m????Unknown
{#HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      @9      @A      @I      @a??u [bR?iɐ??????Unknown
?$HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @aHt?3+jM?i??Ɛ????Unknown
e%Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @aHt?3+jM?i???T???Unknown?
v&HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @aHt?3+jM?i`?`?????Unknown
?'HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aHt?3+jM?i=?-1	???Unknown
a(HostCast"sequential/Cast(1      @9      @A      @I      @aHt?3+jM?i|??c#???Unknown
?)HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aHt?3+jM?i?w?F?*???Unknown
t*Host_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @aHt?3+jM?i?s??2???Unknown
?+HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      @9      @A      @I      @aHt?3+jM?i?oa\s9???Unknown
o,HostSigmoid"sequential/dense_2/Sigmoid(1      @9      @A      @I      @aHt?3+jM?i?k.??@???Unknown
t-HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a6??f?F?it(H?QF???Unknown
v.HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a6??f?F?iZ?a??K???Unknown
\/HostGreater"Greater(1      @9      @A      @I      @a6??f?F?i@?{?YQ???Unknown
u0HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a6??f?F?i&_???V???Unknown
v1HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a6??f?F?i?oa\???Unknown
x2HostCast"&gradient_tape/binary_crossentropy/Cast(1      @9      @A      @I      @a6??f?F?i???W?a???Unknown
?3HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a6??f?F?iؕ??ig???Unknown
?4HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a6??f?F?i?R?'?l???Unknown
?5Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a6??f?F?i?qr???Unknown
?6HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a6??f?F?i??/??w???Unknown
?7HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a6??f?F?ip?I?x}???Unknown
?8HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a6??f?F?iVFc??????Unknown
v9HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @aHt?3+j=?iE???????Unknown
v:HostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @aHt?3+j=?i4B0SW????Unknown
X;HostCast"Cast_2(1       @9       @A       @I       @aHt?3+j=?i#???????Unknown
X<HostEqual"Equal(1       @9       @A       @I       @aHt?3+j=?i>?ݱ????Unknown
s=HostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @aHt?3+j=?i?c#_????Unknown
?>HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1       @9       @A       @I       @aHt?3+j=?i?9?h????Unknown
d?HostAddN"SGD/gradients/AddN(1       @9       @A       @I       @aHt?3+j=?i߷0??????Unknown
V@HostSum"Sum_2(1       @9       @A       @I       @aHt?3+j=?i?5??f????Unknown
rAHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @aHt?3+j=?i???8????Unknown
vBHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @aHt?3+j=?i?1d~?????Unknown
zCHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1       @9       @A       @I       @aHt?3+j=?i????n????Unknown
~DHostSelect"*binary_crossentropy/logistic_loss/Select_1(1       @9       @A       @I       @aHt?3+j=?i?-1	????Unknown
vEHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @aHt?3+j=?iy??Nɲ???Unknown
?FHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @aHt?3+j=?ih)??v????Unknown
uGHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @aHt?3+j=?iW?d?#????Unknown
bHHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @aHt?3+j=?iF%?ѽ???Unknown
wIHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @aHt?3+j=?i5?1d~????Unknown
~JHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @aHt?3+j=?i$!??+????Unknown
?KHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @aHt?3+j=?i????????Unknown
?LHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @aHt?3+j=?ie4?????Unknown
?MHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @aHt?3+j=?i???y3????Unknown
?NHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @aHt?3+j=?i?2??????Unknown
?OHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @aHt?3+j=?iϖ??????Unknown
~PHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @aHt?3+j=?i??I;????Unknown
?QHostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1       @9       @A       @I       @aHt?3+j=?i??e??????Unknown
?RHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1       @9       @A       @I       @aHt?3+j=?i??ԕ????Unknown
?SHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1       @9       @A       @I       @aHt?3+j=?i??2C????Unknown
vTHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??aHt?3+j-?i????????Unknown
XUHostCast"Cast_3(1      ??9      ??A      ??I      ??aHt?3+j-?iy?_?????Unknown
XVHostCast"Cast_4(1      ??9      ??A      ??I      ??aHt?3+j-?ipKL?????Unknown
TWHostMul"Mul(1      ??9      ??A      ??I      ??aHt?3+j-?ig????????Unknown
|XHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??aHt?3+j-?i^ɲGt????Unknown
|YHostSelect"(binary_crossentropy/logistic_loss/Select(1      ??9      ??A      ??I      ??aHt?3+j-?iUf?J????Unknown
}ZHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??aHt?3+j-?iLG?!????Unknown
w[HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??aHt?3+j-?iC??/?????Unknown
y\HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??aHt?3+j-?i:???????Unknown
?]HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??aHt?3+j-?i13u?????Unknown
?^Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??aHt?3+j-?i(C?|????Unknown
?_HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??aHt?3+j-?i???R????Unknown
}`HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      ??9      ??A      ??I      ??aHt?3+j-?i?L])????Unknown
aHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1      ??9      ??A      ??I      ??aHt?3+j-?i     ???Unknown
4bHostIdentity"Identity(i     ???Unknown?
icHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i     ???Unknown
WdHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i     ???Unknown
WeHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(i     ???Unknown
[fHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i     ???Unknown
YgHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i     ???Unknown
[hHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i     ???Unknown*?b
sHostDataset"Iterator::Model::ParallelMapV2(1     ?b@9     ?b@A     ?b@I     ?b@ar??q??ir??q???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     @_@9     @_@A     @_@I     @_@a??RJ)???i?V?he????Unknown
dHostDataset"Iterator::Model(1     ?h@9     ?h@A      I@I      I@a?B!???i??ƪm????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?@@9     ?@@A     ?@@I     ?@@a??\!ͥ?iga?f???Unknown?
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1     ?@@9     ?@@A     ?@@I     ?@@a??\!ͥ?i?? ????Unknown
VHostMean"Mean(1      6@9      6@A      6@I      6@aс??i?p?g????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      .@9      .@A      .@I      .@a?=?ѓ?i?Qo?????Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      ,@9      ,@A      ,@I      ,@a??'???i,?"?*???Unknown
V	HostCast"Cast(1      &@9      &@A      &@I      &@aс??i?r?)7????Unknown
i
HostWriteSummary"WriteSummary(1      &@9      &@A      &@I      &@aс??i?71}???Unknown?
?HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      &@9      &@A      &@I      &@aс??i?3|8Ç???Unknown
`HostDivNoNan"
div_no_nan(1      &@9      &@A      &@I      &@aс??i????	????Unknown
?Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      &@9      &@A      &@I      &@aс??ip?GOp???Unknown
?HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      &@9      &@A      &@I      &@aс??i?TIN?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @a$@R$??i9U?S%9???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @a$@R$??i?U?X?????Unknown
oHost_FusedMatMul"sequential/dense/Relu(1       @9       @A       @I       @a$@R$??i?U$^E????Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a??'???i-6?bC,???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a??'???iwdgAv???Unknown
vHostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a??'???i??l?????Unknown
?HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      @9      @A      @I      @a??'???iףp=
???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @a??'???iU?Cu;T???Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a??'???i???y9????Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @a??'???i?w?~7????Unknown
lHostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a?`???i(8z??'???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?`???ig?p?g???Unknown
jHostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a?`???i??g?{????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @a?`???i?x^??????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?`???i$9U?S%???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      @9      @A      @I      @a?`???ic?K??d???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @amЦmz?i?????????Unknown
? HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @amЦmz?i?9??s????Unknown
?!HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @amЦmz?i?4?M???Unknown
{"HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      @9      @A      @I      @amЦmz?i7z??'8???Unknown
?#HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a$@R$u?ia?&?ob???Unknown
e$Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a$@R$u?i?z˨?????Unknown?
v%HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a$@R$u?i??o??????Unknown
?&HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a$@R$u?i?z?G????Unknown
a'HostCast"sequential/Cast(1      @9      @A      @I      @a$@R$u?i	???????Unknown
?(HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a$@R$u?i3{]??5???Unknown
t)Host_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @a$@R$u?i]??`???Unknown
?*HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      @9      @A      @I      @a$@R$u?i?{??g????Unknown
o+HostSigmoid"sequential/dense_2/Sigmoid(1      @9      @A      @I      @a$@R$u?i??J??????Unknown
t,HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?`??o?i?[F?e????Unknown
v-HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?`??o?i??A?????Unknown
\.HostGreater"Greater(1      @9      @A      @I      @a?`??o?i=?????Unknown
u/HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?`??o?i1|8Ç3???Unknown
v0HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?`??o?iQ?3?=S???Unknown
x1HostCast"&gradient_tape/binary_crossentropy/Cast(1      @9      @A      @I      @a?`??o?iq</??r???Unknown
?2HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?`??o?i??*ɩ????Unknown
?3HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?`??o?i??%?_????Unknown
?4Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a?`??o?i?\!?????Unknown
?5HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?`??o?i????????Unknown
?6HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?`??o?iс???Unknown
?7HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?`??o?i1}?71???Unknown
v8HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a$@R$e?iF?e?[F???Unknown
v9HostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a$@R$e?i[???[???Unknown
X:HostCast"Cast_2(1       @9       @A       @I       @a$@R$e?ip=
ףp???Unknown
X;HostEqual"Equal(1       @9       @A       @I       @a$@R$e?i?}\?ǅ???Unknown
s<HostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a$@R$e?i?????????Unknown
?=HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1       @9       @A       @I       @a$@R$e?i?? ?????Unknown
d>HostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a$@R$e?i?=S?3????Unknown
V?HostSum"Sum_2(1       @9       @A       @I       @a$@R$e?i?}??W????Unknown
r@HostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a$@R$e?i????{????Unknown
vAHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a$@R$e?i?I?????Unknown
zBHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1       @9       @A       @I       @a$@R$e?i>??????Unknown
~CHostSelect"*binary_crossentropy/logistic_loss/Select_1(1       @9       @A       @I       @a$@R$e?i-~???.???Unknown
vDHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a$@R$e?iB?@?D???Unknown
?EHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a$@R$e?iW???/Y???Unknown
uFHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a$@R$e?il>??Sn???Unknown
bGHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a$@R$e?i?~7?w????Unknown
wHHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a$@R$e?i???雘???Unknown
~IHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a$@R$e?i???꿭???Unknown
?JHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a$@R$e?i?>.??????Unknown
?KHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a$@R$e?i?~??????Unknown
?LHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a$@R$e?i????+????Unknown
?MHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a$@R$e?i??$?O???Unknown
?NHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a$@R$e?i?w?s???Unknown
~OHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a$@R$e?i)???,???Unknown
?PHostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1       @9       @A       @I       @a$@R$e?i>???A???Unknown
?QHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1       @9       @A       @I       @a$@R$e?iS?m??V???Unknown
?RHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1       @9       @A       @I       @a$@R$e?ih???l???Unknown
vSHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a$@R$U?is_i??v???Unknown
XTHostCast"Cast_3(1      ??9      ??A      ??I      ??a$@R$U?i~?'????Unknown
XUHostCast"Cast_4(1      ??9      ??A      ??I      ??a$@R$U?i?????????Unknown
TVHostMul"Mul(1      ??9      ??A      ??I      ??a$@R$U?i??d?K????Unknown
|WHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a$@R$U?i???ݠ???Unknown
|XHostSelect"(binary_crossentropy/logistic_loss/Select(1      ??9      ??A      ??I      ??a$@R$U?i????o????Unknown
}YHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a$@R$U?i?`?????Unknown
wZHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a$@R$U?i??	??????Unknown
y[HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a$@R$U?i?_??%????Unknown
?\HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a$@R$U?i?[??????Unknown
?]Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a$@R$U?i???I????Unknown
?^HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a$@R$U?i쿭??????Unknown
}_HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      ??9      ??A      ??I      ??a$@R$U?i??V?m????Unknown
`HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1      ??9      ??A      ??I      ??a$@R$U?i     ???Unknown
4aHostIdentity"Identity(i     ???Unknown?
ibHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i     ???Unknown
WcHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i     ???Unknown
WdHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(i     ???Unknown
[eHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i     ???Unknown
YfHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i     ???Unknown
[gHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i     ???Unknown2CPU