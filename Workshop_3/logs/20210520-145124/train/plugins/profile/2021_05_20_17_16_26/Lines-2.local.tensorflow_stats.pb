"?d
BHostIDLE"IDLE1     o?@A     o?@a??7????i??7?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?@9     ?@A     ?@I     ?@a]e՘?O??iazRO\???Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      F@9      F@A      F@I      F@aT????q?i???BI???Unknown
iHostWriteSummary"WriteSummary(1      <@9      <@A      <@I      <@a?J?Mqf?i?2R?????Unknown?
uHost_FusedMatMul"sequential_20/dense_63/Relu(1      9@9      9@A      9@I      9@a??X??	d?iʋ??é???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      6@9      6@A      6@I      6@aT????a?i`?\?e????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      4@9      4@A      4@I      4@aZz7?`?i??mm????Unknown
uHost_FusedMatMul"sequential_20/dense_64/Relu(1      1@9      1@A      1@I      1@a??O?$@[?i?4?????Unknown
?	HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      .@9      .@A      .@I      .@a-?7?kX?i~??5????Unknown
d
HostDataset"Iterator::Model(1      @@9      @@A      (@I      (@a?ҒBV<S?i?a?????Unknown
?HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      (@9      (@A      (@I      (@a?ҒBV<S?iPc/?O????Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@aZz7?P?i} ?ZS ???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      "@9      "@A      "@I      "@a?;?c??L?i?$?????Unknown
?HostMatMul"-gradient_tape/sequential_20/dense_64/MatMul_1(1      "@9      "@A      "@I      "@a?;?c??L?i?}?????Unknown
gHostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@a?;?c??L?i??;????Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @ac??XȥI?i?6??`???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1       @9       @A       @I       @ac??XȥI?i?g ?"???Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @ac??XȥI?i}??3)???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @ac??XȥI?in?.?/???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @ac??XȥI?i_?Dv6???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @ac??XȥI?iP+[?o<???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1       @9       @A       @I       @ac??XȥI?iA\qZ?B???Unknown
HostMatMul"+gradient_tape/sequential_20/dense_65/MatMul(1       @9       @A       @I       @ac??XȥI?i2???BI???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a?J?MqF?i?Z?N???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a?J?MqF?i?b.T{T???Unknown
HostMatMul"+gradient_tape/sequential_20/dense_63/MatMul(1      @9      @A      @I      @a?J?MqF?i???Z???Unknown
HostMatMul"+gradient_tape/sequential_20/dense_64/MatMul(1      @9      @A      @I      @a?J?MqF?i~8?۳_???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?ҒBV<C?i3?e??d???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a?ҒBV<C?i???Ri???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @aZz7?@?i`D?Sm???Unknown
vHostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @aZz7?@?i???Uq???Unknown
~ HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @aZz7?@?i???Wu???Unknown
?!HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @aZz7?@?iD?-?Yy???Unknown
?"HostBiasAddGrad"8gradient_tape/sequential_20/dense_63/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aZz7?@?i??{?[}???Unknown
?#HostBiasAddGrad"8gradient_tape/sequential_20/dense_64/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aZz7?@?ir??r]????Unknown
?$HostMatMul"-gradient_tape/sequential_20/dense_65/MatMul_1(1      @9      @A      @I      @aZz7?@?i	?Z_????Unknown
x%Host_FusedMatMul"sequential_20/dense_65/BiasAdd(1      @9      @A      @I      @aZz7?@?i?veAa????Unknown
v&HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @ac??Xȥ9?i?p??????Unknown
v'HostAssignAddVariableOp"AssignAddVariableOp_3(1      @9      @A      @I      @ac??Xȥ9?i??{?ʏ???Unknown
V(HostMean"Mean(1      @9      @A      @I      @ac??Xȥ9?i??l?????Unknown
s)HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @ac??Xȥ9?i?ؑ%4????Unknown
z*HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @ac??Xȥ9?i????h????Unknown
?+Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @ac??Xȥ9?ip	???????Unknown
?,HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @ac??Xȥ9?i?!?Pҟ???Unknown
?-HostBiasAddGrad"8gradient_tape/sequential_20/dense_65/BiasAdd/BiasAddGrad(1      @9      @A      @I      @ac??Xȥ9?i`:?	????Unknown
?.HostReadVariableOp"-sequential_20/dense_63/BiasAdd/ReadVariableOp(1      @9      @A      @I      @ac??Xȥ9?i?R??;????Unknown
?/HostReadVariableOp"-sequential_20/dense_64/BiasAdd/ReadVariableOp(1      @9      @A      @I      @ac??Xȥ9?iPk?{p????Unknown
?0HostReadVariableOp",sequential_20/dense_64/MatMul/ReadVariableOp(1      @9      @A      @I      @ac??Xȥ9?iȃ?4?????Unknown
s1HostSigmoid"sequential_20/dense_65/Sigmoid(1      @9      @A      @I      @ac??Xȥ9?i@???ٯ???Unknown
t2HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?ҒBV<3?i???xA????Unknown
v3HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?ҒBV<3?i?@{?????Unknown
X4HostCast"Cast_3(1      @9      @A      @I      @a?ҒBV<3?iN?C?????Unknown
\5HostGreater"Greater(1      @9      @A      @I      @a?ҒBV<3?i??x????Unknown
?6HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?ҒBV<3?i8ԣ߻???Unknown
j7HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a?ҒBV<3?i\??.G????Unknown
r8HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a?ҒBV<3?i??d??????Unknown
v9HostExp"%binary_crossentropy/logistic_loss/Exp(1      @9      @A      @I      @a?ҒBV<3?i/-D????Unknown
v:HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?ҒBV<3?ij???}????Unknown
v;HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?ҒBV<3?i?ӽY?????Unknown
~<HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a?ҒBV<3?i&??L????Unknown
?=HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a?ҒBV<3?ixxNo?????Unknown
v>HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @ac??Xȥ)?i???N????Unknown
V?HostCast"Cast(1       @9       @A       @I       @ac??Xȥ)?i??Y(?????Unknown
X@HostCast"Cast_5(1       @9       @A       @I       @ac??Xȥ)?i,߄?????Unknown
XAHostEqual"Equal(1       @9       @A       @I       @ac??Xȥ)?ih?d?????Unknown
TBHostMul"Mul(1       @9       @A       @I       @ac??Xȥ)?i?5?=?????Unknown
uCHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @ac??Xȥ)?i??o?R????Unknown
|DHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @ac??Xȥ)?iN???????Unknown
dEHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @ac??Xȥ)?iX?zS?????Unknown
vFHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @ac??Xȥ)?i?f ?!????Unknown
?GHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @ac??Xȥ)?i????????Unknown
}HHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @ac??Xȥ)?iiV????Unknown
`IHostDivNoNan"
div_no_nan(1       @9       @A       @I       @ac??Xȥ)?iH???????Unknown
uJHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @ac??Xȥ)?i??"?????Unknown
wKHostReadVariableOp"div_no_nan/ReadVariableOp_1(1       @9       @A       @I       @ac??Xȥ)?i?#?~%????Unknown
bLHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @ac??Xȥ)?i??!ۿ????Unknown
?MHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @ac??Xȥ)?i8<?7Z????Unknown
?NHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @ac??Xȥ)?it?,??????Unknown
?OHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @ac??Xȥ)?i?T???????Unknown
?PHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @ac??Xȥ)?i??7M)????Unknown
~QHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @ac??Xȥ)?i(m???????Unknown
?RHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @ac??Xȥ)?id?B^????Unknown
?SHostReluGrad"-gradient_tape/sequential_20/dense_63/ReluGrad(1       @9       @A       @I       @ac??Xȥ)?i???b?????Unknown
?THostReluGrad"-gradient_tape/sequential_20/dense_64/ReluGrad(1       @9       @A       @I       @ac??Xȥ)?i?N??????Unknown
?UHostReadVariableOp"-sequential_20/dense_65/BiasAdd/ReadVariableOp(1       @9       @A       @I       @ac??Xȥ)?i??-????Unknown
?VHostReadVariableOp",sequential_20/dense_65/MatMul/ReadVariableOp(1       @9       @A       @I       @ac??Xȥ)?iT*Yx?????Unknown
XWHostCast"Cast_4(1      ??9      ??A      ??I      ??ac??Xȥ?ir𛦔????Unknown
aXHostIdentity"Identity(1      ??9      ??A      ??I      ??ac??Xȥ?i????a????Unknown?
yYHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??ac??Xȥ?i?|!/????Unknown
xZHostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??ac??Xȥ?i?Bd1?????Unknown
?[HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??ac??Xȥ?i??_?????Unknown
?\Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??ac??Xȥ?i?鍖????Unknown
?]HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      ??9      ??A      ??I      ??ac??Xȥ?i&?,?c????Unknown
?^HostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??ac??Xȥ?iD[o?0????Unknown
?_HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??ac??Xȥ?ib!??????Unknown
?`HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??ac??Xȥ?i???F?????Unknown
?aHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??ac??Xȥ?i??7u?????Unknown
?bHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??ac??Xȥ?i?sz?e????Unknown
?cHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??ac??Xȥ?i?9??2????Unknown
?dHostReadVariableOp",sequential_20/dense_63/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??ac??Xȥ?i?????????Unknown
ieHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i?????????Unknown
JfHostReadVariableOp"div_no_nan_1/ReadVariableOp(i?????????Unknown
WgHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i?????????Unknown
YhHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown*?d
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?@9     ?@A     ?@I     ?@a?0g?>??i?0g?>???Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      F@9      F@A      F@I      F@aW?]ie??i9B?@?#???Unknown
iHostWriteSummary"WriteSummary(1      <@9      <@A      <@I      <@a@bw?#??iKM??????Unknown?
uHost_FusedMatMul"sequential_20/dense_63/Relu(1      9@9      9@A      9@I      9@aK<A??ē?i-WI??r???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      6@9      6@A      6@I      6@aW?]ie??i?_5?????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      4@9      4@A      4@I      4@a?????i?g?`?|???Unknown
uHost_FusedMatMul"sequential_20/dense_64/Relu(1      1@9      1@A      1@I      1@a)???????i?n?????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      .@9      .@A      .@I      .@a?{?ո??iot??F???Unknown
d	HostDataset"Iterator::Model(1      @@9      @@A      (@I      (@a?/fD???i.y???????Unknown
?
HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      (@9      (@A      (@I      (@a?/fD???i?}M?????Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@a????iၡA???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      "@9      "@A      "@I      "@av??fw|?ip???V???Unknown
?HostMatMul"-gradient_tape/sequential_20/dense_64/MatMul_1(1      "@9      "@A      "@I      "@av??fw|?i????????Unknown
gHostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@av??fw|?i??7??????Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a۔??My?i??G
t????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1       @9       @A       @I       @a۔??My?i??Wk.???Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a۔??My?i?g̪`???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @a۔??My?i6?w-F????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @a۔??My?i`????????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a۔??My?i????|????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1       @9       @A       @I       @a۔??My?i???P+???Unknown
HostMatMul"+gradient_tape/sequential_20/dense_65/MatMul(1       @9       @A       @I       @a۔??My?iޥ???]???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a@bw?#v?i?????????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a@bw?#v?if???C????Unknown
HostMatMul"+gradient_tape/sequential_20/dense_63/MatMul(1      @9      @A      @I      @a@bw?#v?i*????????Unknown
HostMatMul"+gradient_tape/sequential_20/dense_64/MatMul(1      @9      @A      @I      @a@bw?#v?i??o?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?/fD?r?iM?;?4???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a?/fD?r?i????Z???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a???o?i????]z???Unknown
vHostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a???o?i??[??????Unknown
~HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a???o?i??ퟹ???Unknown
? HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a???o?i???	A????Unknown
?!HostBiasAddGrad"8gradient_tape/sequential_20/dense_63/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???o?i??Y&?????Unknown
?"HostBiasAddGrad"8gradient_tape/sequential_20/dense_64/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???o?i??C????Unknown
?#HostMatMul"-gradient_tape/sequential_20/dense_65/MatMul_1(1      @9      @A      @I      @a???o?i?í_$8???Unknown
x$Host_FusedMatMul"sequential_20/dense_65/BiasAdd(1      @9      @A      @I      @a???o?i|?W|?W???Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a۔??Mi?i??,q???Unknown
v&HostAssignAddVariableOp"AssignAddVariableOp_3(1      @9      @A      @I      @a۔??Mi?i??g?`????Unknown
V'HostMean"Mean(1      @9      @A      @I      @a۔??Mi?i;?????Unknown
s(HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a۔??Mi?i??w>?????Unknown
z)HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a۔??Mi?ie???I????Unknown
?*Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a۔??Mi?i?·??????Unknown
?+HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a۔??Mi?i??P????Unknown
?,HostBiasAddGrad"8gradient_tape/sequential_20/dense_65/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a۔??Mi?i$җ 3"???Unknown
?-HostReadVariableOp"-sequential_20/dense_63/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a۔??Mi?i????;???Unknown
?.HostReadVariableOp"-sequential_20/dense_64/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a۔??Mi?iNէa?T???Unknown
?/HostReadVariableOp",sequential_20/dense_64/MatMul/ReadVariableOp(1      @9      @A      @I      @a۔??Mi?i??/n???Unknown
s0HostSigmoid"sequential_20/dense_65/Sigmoid(1      @9      @A      @I      @a۔??Mi?ixط?i????Unknown
t1HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?/fD?b?i??d????Unknown
v2HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?/fD?b?i?ڃK^????Unknown
X3HostCast"Cast_3(1      @9      @A      @I      @a?/fD?b?i???X????Unknown
\4HostGreater"Greater(1      @9      @A      @I      @a?/fD?b?i8?O?R????Unknown
?5HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?/fD?b?ih޵M????Unknown
j6HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a?/fD?b?i??]G????Unknown
r7HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a?/fD?b?i????A???Unknown
v8HostExp"%binary_crossentropy/logistic_loss/Exp(1      @9      @A      @I      @a?/fD?b?i????;???Unknown
v9HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?/fD?b?i(?M*62???Unknown
v:HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?/fD?b?iX??n0E???Unknown
~;HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a?/fD?b?i???*X???Unknown
?<HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a?/fD?b?i???$k???Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a۔??MY?i?????w???Unknown
V>HostCast"Cast(1       @9       @A       @I       @a۔??MY?iL??r????Unknown
X?HostCast"Cast_5(1       @9       @A       @I       @a۔??MY?i?K?????Unknown
X@HostEqual"Equal(1       @9       @A       @I       @a۔??MY?i???X?????Unknown
TAHostMul"Mul(1       @9       @A       @I       @a۔??MY?i???0g????Unknown
uBHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a۔??MY?it?	????Unknown
|CHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a۔??MY?i>?[??????Unknown
dDHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a۔??MY?iퟹ[????Unknown
vEHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a۔??MY?i????????Unknown
?FHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a۔??MY?i??'j?????Unknown
}GHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a۔??MY?if?kBP????Unknown
`HHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a۔??MY?i0??????Unknown
uIHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a۔??MY?i????????Unknown
wJHostReadVariableOp"div_no_nan/ReadVariableOp_1(1       @9       @A       @I       @a۔??MY?i??7?D???Unknown
bKHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a۔??MY?i??{??(???Unknown
?LHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a۔??MY?iX??{?5???Unknown
?MHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a۔??MY?i"?T9B???Unknown
?NHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a۔??MY?i??G,?N???Unknown
?OHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a۔??MY?i????[???Unknown
~PHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a۔??MY?i????-h???Unknown
?QHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a۔??MY?iJ???t???Unknown
?RHostReluGrad"-gradient_tape/sequential_20/dense_63/ReluGrad(1       @9       @A       @I       @a۔??MY?i?W?{????Unknown
?SHostReluGrad"-gradient_tape/sequential_20/dense_64/ReluGrad(1       @9       @A       @I       @a۔??MY?i???e"????Unknown
?THostReadVariableOp"-sequential_20/dense_65/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a۔??MY?i???=ɚ???Unknown
?UHostReadVariableOp",sequential_20/dense_65/MatMul/ReadVariableOp(1       @9       @A       @I       @a۔??MY?ir?#p????Unknown
XVHostCast"Cast_4(1      ??9      ??A      ??I      ??a۔??MI?i??E?í???Unknown
aWHostIdentity"Identity(1      ??9      ??A      ??I      ??a۔??MI?i<?g?????Unknown?
yXHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a۔??MI?i???Zj????Unknown
xYHostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??a۔??MI?i??ƽ????Unknown
?ZHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a۔??MI?ik??2????Unknown
?[Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a۔??MI?i????d????Unknown
?\HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      ??9      ??A      ??I      ??a۔??MI?i5??????Unknown
?]HostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??a۔??MI?i??3w????Unknown
?^HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a۔??MI?i??U?^????Unknown
?_HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a۔??MI?id?wO?????Unknown
?`HostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a۔??MI?i????????Unknown
?aHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a۔??MI?i.??'Y????Unknown
?bHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a۔??MI?i??ݓ?????Unknown
?cHostReadVariableOp",sequential_20/dense_63/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a۔??MI?i?????????Unknown
idHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i?????????Unknown
JeHostReadVariableOp"div_no_nan_1/ReadVariableOp(i?????????Unknown
WfHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i?????????Unknown
YgHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown2CPU