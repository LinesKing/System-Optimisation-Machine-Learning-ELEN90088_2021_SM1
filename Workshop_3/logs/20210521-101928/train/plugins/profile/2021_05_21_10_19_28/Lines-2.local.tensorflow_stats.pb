"?d
BHostIDLE"IDLE1     ?@A     ?@ag<:ޗ???ig<:ޗ????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     ?Z@9     ?Z@A     ?Z@I     ?Z@a?㒯???i?X?Z-????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?U@9     ?U@A     ?U@I     ?U@a??)A???i=y d6???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      C@9      C@A      C@I      C@a?Wd?	t??i?
???w???Unknown?
vHostSum"%binary_crossentropy/weighted_loss/Sum(1     ?@@9     ?@@A     ?@@I     ?@@aX:Ɂ??|?i???????Unknown
?HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      :@9      :@A      :@I      :@aɖ?׃v?i??p?????Unknown
lHostIteratorGetNext"IteratorGetNext(1      8@9      8@A      8@I      8@aW?c^x?t?i??-w????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      :@9      :@A      6@I      6@a??0s?i??/??-???Unknown
?	HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      4@9      4@A      4@I      4@a?????Qq?i??wmP???Unknown
{
HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      1@9      1@A      1@I      1@a<?b0Uqm?iFR?q?m???Unknown
dHostDataset"Iterator::Model(1      ^@9      ^@A      ,@I      ,@a???7?h?i??????Unknown
oHost_FusedMatMul"sequential/dense/Relu(1      ,@9      ,@A      ,@I      ,@a???7?h?i????\????Unknown
iHostWriteSummary"WriteSummary(1      *@9      *@A      *@I      *@aɖ?׃f?i?|???????Unknown?
qHost_FusedMatMul"sequential/dense_1/Relu(1      *@9      *@A      *@I      *@aɖ?׃f?ibQ?d????Unknown
VHostCast"Cast(1      (@9      (@A      (@I      (@aW?c^x?d?i/w?-????Unknown
zHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      &@9      &@A      &@I      &@a??0c?i?? :????Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      $@9      $@A      $@I      $@a?????Qa?iץTڋ???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a????,_?i?p?4"???Unknown
`HostGatherV2"
GatherV2_1(1       @9       @A       @I       @at?/???[?i??/?!???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1       @9       @A       @I       @at?/???[?im?n*?/???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @at?/???[?iK8X%?=???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1       @9       @A       @I       @at?/???[?i)?A ?K???Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @at?/???[?ih+iY???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a???7?X?i?̷??e???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a???7?X?i?1DR?q???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a???7?X?i?????}???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a???7?X?i??\??????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a???7?X?iq`?$????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @a???7?X?iS?u?&????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      @9      @A      @I      @a???7?X?i5*\F????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???7?X?i???e????Unknown
? HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @aW?c^x?T?i???3?????Unknown
?!HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @aW?c^x?T?i???o.????Unknown
t"Host_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @aW?c^x?T?i?$??????Unknown
?#HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?????QQ?i?#??;????Unknown
?$HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?????QQ?i?"?e?????Unknown
~%HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?????QQ?i?!?B?????Unknown
?&HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?????QQ?ix d6????Unknown
?'HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?????QQ?ic6?????Unknown
v(HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @at?/???K?iR??y????Unknown
X)HostCast"Cast_4(1      @9      @A      @I      @at?/???K?iA??????Unknown
?*HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @at?/???K?i0??t????Unknown
e+Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @at?/???K?iO	?? ???Unknown?
V,HostMean"Mean(1      @9      @A      @I      @at?/???K?i~o?'???Unknown
v-HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @at?/???K?i????o.???Unknown
?.HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @at?/???K?i??gj]5???Unknown
?/HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @at?/???K?i?~??J<???Unknown
?0HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @at?/???K?i?JQe8C???Unknown
}1HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @at?/???K?i???%J???Unknown
?2HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      @9      @A      @I      @at?/???K?i??:`Q???Unknown
t3HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @aW?c^x?D?i?{R~EV???Unknown
v4HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @aW?c^x?D?i?j?w[???Unknown
X5HostCast"Cast_3(1      @9      @A      @I      @aW?c^x?D?i?????`???Unknown
\6HostGreater"Greater(1      @9      @A      @I      @aW?c^x?D?itF???e???Unknown
d7HostAddN"SGD/gradients/AddN(1      @9      @A      @I      @aW?c^x?D?ig߰?k???Unknown
r8HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @aW?c^x?D?iZx?@p???Unknown
?9HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @aW?c^x?D?iM?2ru???Unknown
v:HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @aW?c^x?D?i@??P?z???Unknown
v;HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @aW?c^x?D?i3Co????Unknown
`<HostDivNoNan"
div_no_nan(1      @9      @A      @I      @aW?c^x?D?i&?&?????Unknown
b=HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @aW?c^x?D?iu>?:????Unknown
?>HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @aW?c^x?D?iV?l????Unknown
??Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @aW?c^x?D?i??m瞔???Unknown
?@Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @aW?c^x?D?i???љ???Unknown
?AHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aW?c^x?D?i?؜#????Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @at?/???;?i?>W?y????Unknown
vCHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @at?/???;?iդ??????Unknown
XDHostCast"Cast_5(1       @9       @A       @I       @at?/???;?i?
?_g????Unknown
XEHostEqual"Equal(1       @9       @A       @I       @at?/???;?i?p?ެ???Unknown
|FHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @at?/???;?i??@?T????Unknown
jGHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @at?/???;?i?<??˳???Unknown
vHHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @at?/???;?i???ZB????Unknown
?IHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @at?/???;?i?p?????Unknown
uJHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @at?/???;?i?n*?/????Unknown
wKHostReadVariableOp"div_no_nan/ReadVariableOp_1(1       @9       @A       @I       @at?/???;?i??䖦????Unknown
wLHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @at?/???;?i?:?U????Unknown
xMHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @at?/???;?i??Y?????Unknown
~NHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @at?/???;?i}?
????Unknown
?OHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @at?/???;?iulΑ?????Unknown
~PHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @at?/???;?im҈P?????Unknown
?QHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @at?/???;?ie8Co????Unknown
}RHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @at?/???;?i]????????Unknown
SHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @at?/???;?iU??\????Unknown
?THostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @at?/???;?iMjrK?????Unknown
oUHostSigmoid"sequential/dense_2/Sigmoid(1       @9       @A       @I       @at?/???;?iE?,
J????Unknown
TVHostMul"Mul(1      ??9      ??A      ??I      ??at?/???+?iA?i????Unknown
sWHostReadVariableOp"SGD/Cast/ReadVariableOp(1      ??9      ??A      ??I      ??at?/???+?i=6???????Unknown
uXHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ??9      ??A      ??I      ??at?/???+?i9iD(|????Unknown
}YHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??at?/???+?i5???7????Unknown
?ZHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??at?/???+?i1????????Unknown
?[HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??at?/???+?i-\F?????Unknown
?\HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??at?/???+?i)5??i????Unknown
?]Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??at?/???+?i%h%????Unknown
?^HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??at?/???+?i!?sd?????Unknown
?_HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??at?/???+?i??Û????Unknown
?`HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??at?/???+?i.#W????Unknown
?aHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??at?/???+?i4??????Unknown
?bHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??at?/???+?ig???????Unknown
?cHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??at?/???+?i?EA?????Unknown
?dHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??at?/???+?i	͢?D????Unknown
?eHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??at?/???+?i     ???Unknown
4fHostIdentity"Identity(i     ???Unknown?
LgHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i     ???Unknown
[hHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i     ???Unknown*?d
sHostDataset"Iterator::Model::ParallelMapV2(1     ?Z@9     ?Z@A     ?Z@I     ?Z@a????
??i????
???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?U@9     ?U@A     ?U@I     ?U@a?????F??iAL? &W???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      C@9      C@A      C@I      C@ag???Qߩ?i? &W????Unknown?
vHostSum"%binary_crossentropy/weighted_loss/Sum(1     ?@@9     ?@@A     ?@@I     ?@@aڨ?l?w??i???
b???Unknown
?HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      :@9      :@A      :@I      :@aG}g?????ir1??????Unknown
lHostIteratorGetNext"IteratorGetNext(1      8@9      8@A      8@I      8@aAL? &W??i??F}g????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      :@9      :@A      6@I      6@ax6?;???ibr1?????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      4@9      4@A      4@I      4@am?w6?;??i??l?w6???Unknown
{	HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      1@9      1@A      1@I      1@a]AL? &??i??Q?٨???Unknown
d
HostDataset"Iterator::Model(1      ^@9      ^@A      ,@I      ,@aL? &W??iR?٨?l???Unknown
oHost_FusedMatMul"sequential/dense/Relu(1      ,@9      ,@A      ,@I      ,@aL? &W??i??
br???Unknown
iHostWriteSummary"WriteSummary(1      *@9      *@A      *@I      *@aG}g?????i? &W????Unknown?
qHost_FusedMatMul"sequential/dense_1/Relu(1      *@9      *@A      *@I      *@aG}g?????i?\AL? ???Unknown
VHostCast"Cast(1      (@9      (@A      (@I      (@aAL? &W??i??F}g????Unknown
zHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      &@9      &@A      &@I      &@ax6?;???i?w6?;???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      $@9      $@A      $@I      $@am?w6?;??i&W?+????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@abr1????i?l?w6????Unknown
`HostGatherV2"
GatherV2_1(1       @9       @A       @I       @aW?+?Ʌ?i1???\A???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1       @9       @A       @I       @aW?+?Ʌ?ir1??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @aW?+?Ʌ?i?Q?٨????Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1       @9       @A       @I       @aW?+?Ʌ?i?????F???Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @aW?+?Ʌ?i5?;?????Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @aL? &W??i?l?w6????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @aL? &W??i??l?w6???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @aL? &W??i`r1?????Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @aL? &W??i????????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @aL? &W??i?w6?;???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @aL? &W??i???F}g???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      @9      @A      @I      @aL? &W??iD}g??????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aL? &W??i?????????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @aAL? &W??i.???\A???Unknown
? HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @aAL? &W??i_r1?????Unknown
t!Host_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @aAL? &W??i?+??????Unknown
?"HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @am?w6?;{?i9???????Unknown
?#HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @am?w6?;{?i?
br1???Unknown
~$HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @am?w6?;{?i???F}g???Unknown
?%HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @am?w6?;{?i4?;?????Unknown
?&HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @am?w6?;{?i?٨?l????Unknown
v'HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @aW?+??u?i?????????Unknown
X(HostCast"Cast_4(1      @9      @A      @I      @aW?+??u?i&W?+???Unknown
?)HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @aW?+??u?i@L? &W???Unknown
e*Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @aW?+??u?iar1?????Unknown?
V+HostMean"Mean(1      @9      @A      @I      @aW?+??u?i??\AL????Unknown
v,HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @aW?+??u?i???Q?????Unknown
?-HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @aW?+??u?i??
br???Unknown
?.HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @aW?+??u?i?
br1???Unknown
?/HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aW?+??u?i1???\???Unknown
}0HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @aW?+??u?i'W?+????Unknown
?1HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      @9      @A      @I      @aW?+??u?iH}g??????Unknown
t2HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @aAL? &Wp?i?٨?l????Unknown
v3HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @aAL? &Wp?iz6?;????Unknown
X4HostCast"Cast_3(1      @9      @A      @I      @aAL? &Wp?i?+?????Unknown
\5HostGreater"Greater(1      @9      @A      @I      @aAL? &Wp?i??l?w6???Unknown
d6HostAddN"SGD/gradients/AddN(1      @9      @A      @I      @aAL? &Wp?iEL? &W???Unknown
r7HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @aAL? &Wp?iި?l?w???Unknown
?8HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @aAL? &Wp?iw1??????Unknown
v9HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @aAL? &Wp?ibr1????Unknown
v:HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @aAL? &Wp?i???Q?????Unknown
`;HostDivNoNan"
div_no_nan(1      @9      @A      @I      @aAL? &Wp?iB???????Unknown
b<HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @aAL? &Wp?i?w6?;???Unknown
?=HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @aAL? &Wp?it?w6?;???Unknown
?>Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @aAL? &Wp?i1???\???Unknown
??Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @aAL? &Wp?i????F}???Unknown
?@HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aAL? &Wp?i??;?????Unknown
vAHostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @aW?+??e?iO}g??????Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @aW?+??e?i_?+?????Unknown
XCHostCast"Cast_5(1       @9       @A       @I       @aW?+??e?io???Q????Unknown
XDHostEqual"Equal(1       @9       @A       @I       @aW?+??e?i6?;????Unknown
|EHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @aW?+??e?i????
???Unknown
jFHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @aW?+??e?i?\AL? ???Unknown
vGHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @aW?+??e?i??l?w6???Unknown
?HHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @aW?+??e?i???\AL???Unknown
uIHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @aW?+??e?i???
b???Unknown
wJHostReadVariableOp"div_no_nan/ReadVariableOp_1(1       @9       @A       @I       @aW?+??e?iߨ?l?w???Unknown
wKHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @aW?+??e?i?;??????Unknown
xLHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @aW?+??e?i??F}g????Unknown
~MHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @aW?+??e?ibr1????Unknown
?NHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @aW?+??e?i????????Unknown
~OHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @aW?+??e?i/???????Unknown
?PHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @aW?+??e?i????????Unknown
}QHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @aW?+??e?iO? &W???Unknown
RHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @aW?+??e?i_AL? &???Unknown
?SHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aW?+??e?io?w6?;???Unknown
oTHostSigmoid"sequential/dense_2/Sigmoid(1       @9       @A       @I       @aW?+??e?ig???Q???Unknown
TUHostMul"Mul(1      ??9      ??A      ??I      ??aW?+??U?i1???\???Unknown
sVHostReadVariableOp"SGD/Cast/ReadVariableOp(1      ??9      ??A      ??I      ??aW?+??U?i???F}g???Unknown
uWHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ??9      ??A      ??I      ??aW?+??U?i??
br???Unknown
}XHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??aW?+??U?i????F}???Unknown
?YHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??aW?+??U?i'W?+????Unknown
?ZHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??aW?+??U?i? &W????Unknown
?[HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??aW?+??U?i7?;?????Unknown
?\Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??aW?+??U?i??Q?٨???Unknown
?]HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??aW?+??U?iG}g??????Unknown
?^HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??aW?+??U?i?F}g?????Unknown
?_HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??aW?+??U?iW?+?????Unknown
?`HostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??aW?+??U?i?٨?l????Unknown
?aHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??aW?+??U?ig???Q????Unknown
?bHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??aW?+??U?i?l?w6????Unknown
?cHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??aW?+??U?iw6?;????Unknown
?dHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??aW?+??U?i?????????Unknown
4eHostIdentity"Identity(i?????????Unknown?
LfHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i?????????Unknown
[gHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i?????????Unknown2CPU