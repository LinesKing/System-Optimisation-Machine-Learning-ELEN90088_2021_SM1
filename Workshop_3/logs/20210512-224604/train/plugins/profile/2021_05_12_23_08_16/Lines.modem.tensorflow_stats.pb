"?e
BHostIDLE"IDLE1    ?l?@A    ?l?@a(1????i(1?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     8?@9     8?@A     8?@I     8?@a?.[O-??i??_??L???Unknown?
iHostWriteSummary"WriteSummary(1      B@9      B@A      B@I      B@abڨ8k?iRq6?g???Unknown?
gHostStridedSlice"strided_slice(1      B@9      B@A      B@I      B@abڨ8k?i?K?nĂ???Unknown
^HostGatherV2"GatherV2(1      @@9      @@A      @@I      @@a?s??h?i(cG/Ț???Unknown
tHost_FusedMatMul"sequential_7/dense_22/Relu(1      @@9      @@A      @@I      @@a?s??h?i?z??˲???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      >@9      >@A      >@I      >@ao?????f?i?0jtO????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      6@9      6@A      6@I      6@a?0g??`?i?`??????Unknown
d	HostDataset"Iterator::Model(1      H@9      H@A      2@I      2@abڨ8[?i??%%T????Unknown
`
HostGatherV2"
GatherV2_1(1      0@9      0@A      0@I      0@a?s??X?i??pV????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      .@9      .@A      .@I      .@ao?????V?i?4?Ǘ????Unknown
VHostMean"Mean(1      ,@9      ,@A      ,@I      ,@a5?T?HU?i???k	???Unknown
~HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      (@9      (@A      (@I      (@a???p?R?i?'1????Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@abڨ8K?iH^[?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@abڨ8K?iᔅ?????Unknown
~HostMatMul"*gradient_tape/sequential_7/dense_24/MatMul(1      "@9      "@A      "@I      "@abڨ8K?iz˯?]&???Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a?s??H?iWQ??^,???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @a?s??H?i4???_2???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @a?s??H?i] ?`8???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a?s??H?i??E?a>???Unknown
~HostMatMul"*gradient_tape/sequential_7/dense_23/MatMul(1       @9       @A       @I       @a?s??H?i?hk?bD???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @a5?T?HE?i?=???I???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a5?T?HE?i?S?N???Unknown
~HostMatMul"*gradient_tape/sequential_7/dense_22/MatMul(1      @9      @A      @I      @a5?T?HE?i.??%%T???Unknown
?HostMatMul",gradient_tape/sequential_7/dense_23/MatMul_1(1      @9      @A      @I      @a5?T?HE?iO???eY???Unknown
tHost_FusedMatMul"sequential_7/dense_23/Relu(1      @9      @A      @I      @a5?T?HE?ip?ʦ^???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a???p?B?iֶ+~'c???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a???p?B?i<?G2?g???Unknown
vHostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a???p?B?i??c?(l???Unknown
?HostBiasAddGrad"7gradient_tape/sequential_7/dense_22/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???p?B?i$???p???Unknown
?HostBiasAddGrad"7gradient_tape/sequential_7/dense_24/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???p?B?inH?N*u???Unknown
w Host_FusedMatMul"sequential_7/dense_24/BiasAdd(1      @9      @A      @I      @a???p?B?i?l??y???Unknown
?!HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a?P???>?i~?Ϙk}???Unknown
?"HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?P???>?i(T?.,????Unknown
?#HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?P???>?i?????????Unknown
z$HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a?P???>?i|;[?????Unknown
~%HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?P???>?i&?-?m????Unknown
?&HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?P???>?i?"E?.????Unknown
?'HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?P???>?iz?\?????Unknown
?(HostReadVariableOp"+sequential_7/dense_22/MatMul/ReadVariableOp(1      @9      @A      @I      @a?P???>?i$
t??????Unknown
t)HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?s??8?i͆+?????Unknown
v*HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?s??8?i ????????Unknown
V+HostSum"Sum_2(1      @9      @A      @I      @a?s??8?i?R??????Unknown
v,HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a?s??8?i????????Unknown
v-HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?s??8?i????????Unknown
`.HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a?s??8?i??䃲????Unknown
?/HostBiasAddGrad"7gradient_tape/sequential_7/dense_23/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?s??8?i?^???????Unknown
?0HostMatMul",gradient_tape/sequential_7/dense_24/MatMul_1(1      @9      @A      @I      @a?s??8?i?!
t?????Unknown
?1HostReadVariableOp",sequential_7/dense_23/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?s??8?i??쳲???Unknown
r2HostSigmoid"sequential_7/dense_24/Sigmoid(1      @9      @A      @I      @a?s??8?ip?/d?????Unknown
v3HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a???p?2?i??=??????Unknown
V4HostCast"Cast(1      @9      @A      @I      @a???p?2?i??K5????Unknown
X5HostEqual"Equal(1      @9      @A      @I      @a???p?2?i	?Yru????Unknown
\6HostGreater"Greater(1      @9      @A      @I      @a???p?2?i<?g̵????Unknown
d7HostAddN"SGD/gradients/AddN(1      @9      @A      @I      @a???p?2?iov&?????Unknown
r8HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a???p?2?i???6????Unknown
v9HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a???p?2?i?&??v????Unknown
?:HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a???p?2?i9?4?????Unknown
?;HostReadVariableOp",sequential_7/dense_22/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a???p?2?i;K???????Unknown
?<HostReadVariableOp"+sequential_7/dense_23/MatMul/ReadVariableOp(1      @9      @A      @I      @a???p?2?in]??7????Unknown
?=HostReadVariableOp"+sequential_7/dense_24/MatMul/ReadVariableOp(1      @9      @A      @I      @a???p?2?i?o?Bx????Unknown
v>HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a?s??(?i??~?????Unknown
X?HostCast"Cast_3(1       @9       @A       @I       @a?s??(?i?2ݺx????Unknown
X@HostCast"Cast_4(1       @9       @A       @I       @a?s??(?i????????Unknown
XAHostCast"Cast_5(1       @9       @A       @I       @a?s??(?i}??2y????Unknown
sBHostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a?s??(?i?V?n?????Unknown
|CHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a?s??(?ik??y????Unknown
jDHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a?s??(?i???????Unknown
vEHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?s??(?iY{#z????Unknown
?FHostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1       @9       @A       @I       @a?s??(?i??_?????Unknown
?GHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?s??(?iG>(?z????Unknown
}HHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a?s??(?i??1??????Unknown
bIHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a?s??(?i5;{????Unknown
wJHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?s??(?i?bDO?????Unknown
yKHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1       @9       @A       @I       @a?s??(?i#?M?{????Unknown
?LHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a?s??(?i?%W??????Unknown
xMHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?s??(?i?`|????Unknown
?NHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a?s??(?i??i??????Unknown
?OHostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a?s??(?i?Is{|????Unknown
?PHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?s??(?iv?|??????Unknown
?QHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a?s??(?i???|????Unknown
?RHost	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1       @9       @A       @I       @a?s??(?idn?/?????Unknown
?SHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a?s??(?i?Ϙk}????Unknown
~THostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?s??(?iR1???????Unknown
?UHostReluGrad",gradient_tape/sequential_7/dense_23/ReluGrad(1       @9       @A       @I       @a?s??(?iɒ??}????Unknown
?VHostReadVariableOp",sequential_7/dense_24/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?s??(?i@???????Unknown
vWHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?s???i???=?????Unknown
aXHostIdentity"Identity(1      ??9      ??A      ??I      ??a?s???i?U?[~????Unknown?
TYHostMul"Mul(1      ??9      ??A      ??I      ??a?s???it?y>????Unknown
uZHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ??9      ??A      ??I      ??a?s???i0?Ǘ?????Unknown
u[HostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a?s???i?g̵?????Unknown
w\HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?s???i???~????Unknown
?]HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?s???id???>????Unknown
?^Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a?s???i z??????Unknown
?_HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a?s???i?*?-?????Unknown
?`HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?s???i???K????Unknown
?aHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?s???iT??i?????Unknown
?bHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a?s???i=???????Unknown
?cHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      ??9      ??A      ??I      ??a?s???i?????????Unknown
?dHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a?s???i????????Unknown
?eHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a?s???iDO???????Unknown
?fHostReluGrad",gradient_tape/sequential_7/dense_22/ReluGrad(1      ??9      ??A      ??I      ??a?s???i      ???Unknown
igHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i      ???Unknown
[hHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i      ???Unknown*?d
uHostFlushSummaryWriter"FlushSummaryWriter(1     8?@9     8?@A     8?@I     8?@a?J??|??i?J??|???Unknown?
iHostWriteSummary"WriteSummary(1      B@9      B@A      B@I      B@a?T???ۘ?i:-?M?B???Unknown?
gHostStridedSlice"strided_slice(1      B@9      B@A      B@I      B@a?T???ۘ?i?l??	???Unknown
^HostGatherV2"GatherV2(1      @@9      @@A      @@I      @@a)??1???i?g??????Unknown
tHost_FusedMatMul"sequential_7/dense_22/Relu(1      @@9      @@A      @@I      @@a)??1???iC??_k???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      >@9      >@A      >@I      >@a?Fr?????iyQx????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      6@9      6@A      6@I      6@axd?a??i??ǝ????Unknown
dHostDataset"Iterator::Model(1      H@9      H@A      2@I      2@a?T???ۈ?i?g????Unknown
`	HostGatherV2"
GatherV2_1(1      0@9      0@A      0@I      0@a)??1???i??-"nF???Unknown
?
HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      .@9      .@A      .@I      .@a?Fr?????i?'J????Unknown
VHostMean"Mean(1      ,@9      ,@A      ,@I      ,@ad?Y?rU??iiUٟ????Unknown
~HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      (@9      (@A      (@I      (@a?8(eb???iK??b?(???Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a?T????x?i?3??Z???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a?T????x?i??H?W????Unknown
~HostMatMul"*gradient_tape/sequential_7/dense_24/MatMul(1      "@9      "@A      "@I      "@a?T????x?iI%x?????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a)??1?v?i5;???????Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @a)??1?v?i!Q>?p???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @a)??1?v?ig???B???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a)??1?v?i?|??n???Unknown
~HostMatMul"*gradient_tape/sequential_7/dense_23/MatMul(1       @9       @A       @I       @a)??1?v?i??g?????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @ad?Y?rUs?iF?ݮ????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @ad?Y?rUs?iC???Y????Unknown
~HostMatMul"*gradient_tape/sequential_7/dense_22/MatMul(1      @9      @A      @I      @ad?Y?rUs?ir?+????Unknown
?HostMatMul",gradient_tape/sequential_7/dense_23/MatMul_1(1      @9      @A      @I      @ad?Y?rUs?i?_?5???Unknown
tHost_FusedMatMul"sequential_7/dense_23/Relu(1      @9      @A      @I      @ad?Y?rUs?i?YtZ\???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?8(eb?p?iAc#9}???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a?8(eb?p?i?????????Unknown
vHostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?8(eb?p?i#??ȿ???Unknown
?HostBiasAddGrad"7gradient_tape/sequential_7/dense_22/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?8(eb?p?i?T???????Unknown
?HostBiasAddGrad"7gradient_tape/sequential_7/dense_24/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?8(eb?p?i?LL???Unknown
wHost_FusedMatMul"sequential_7/dense_24/BiasAdd(1      @9      @A      @I      @a?8(eb?p?iv?7#???Unknown
? HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a??????k?i*???>???Unknown
?!HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a??????k?i??YtZ???Unknown
?"HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a??????k?i???v???Unknown
z#HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a??????k?iF???????Unknown
~$HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a??????k?i??EP????Unknown
?%HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a??????k?i??
??????Unknown
?&HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a??????k?ibu??????Unknown
?'HostReadVariableOp"+sequential_7/dense_22/MatMul/ReadVariableOp(1      @9      @A      @I      @a??????k?ic1, ???Unknown
t(HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a)??1?f?i?7?D???Unknown
v)HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a)??1?f?iyi7],???Unknown
V*HostSum"Sum_2(1      @9      @A      @I      @a)??1?f?i???uB???Unknown
v+HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a)??1?f?i???=?X???Unknown
v,HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a)??1?f?i????n???Unknown
`-HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a)??1?f?iڤ/D?????Unknown
?.HostBiasAddGrad"7gradient_tape/sequential_7/dense_23/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a)??1?f?i?/a?ך???Unknown
?/HostMatMul",gradient_tape/sequential_7/dense_24/MatMul_1(1      @9      @A      @I      @a)??1?f?iƺ?J?????Unknown
?0HostReadVariableOp",sequential_7/dense_23/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a)??1?f?i?E??????Unknown
r1HostSigmoid"sequential_7/dense_24/Sigmoid(1      @9      @A      @I      @a)??1?f?i???P!????Unknown
v2HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?8(eb?`?i??Z??????Unknown
V3HostCast"Cast(1      @9      @A      @I      @a?8(eb?`?i$!?F????Unknown
X4HostEqual"Equal(1      @9      @A      @I      @a?8(eb?`?i]I%x????Unknown
\5HostGreater"Greater(1      @9      @A      @I      @a?8(eb?`?i?q??j???Unknown
d6HostAddN"SGD/gradients/AddN(1      @9      @A      @I      @a?8(eb?`?iϙ?<?/???Unknown
r7HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a?8(eb?`?i?T??@???Unknown
v8HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a?8(eb?`?iA??"Q???Unknown
?9HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a?8(eb?`?izd?a???Unknown
?:HostReadVariableOp",sequential_7/dense_22/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?8(eb?`?i?:??Fr???Unknown
?;HostReadVariableOp"+sequential_7/dense_23/MatMul/ReadVariableOp(1      @9      @A      @I      @a?8(eb?`?i?b?(ق???Unknown
?<HostReadVariableOp"+sequential_7/dense_24/MatMul/ReadVariableOp(1      @9      @A      @I      @a?8(eb?`?i%?N?k????Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a)??1?V?i?P??w????Unknown
X>HostCast"Cast_3(1       @9       @A       @I       @a)??1?V?i??????Unknown
X?HostCast"Cast_4(1       @9       @A       @I       @a)??1?V?i??P?????Unknown
X@HostCast"Cast_5(1       @9       @A       @I       @a)??1?V?i????????Unknown
sAHostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a)??1?V?i?fJӨ????Unknown
|BHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a)??1?V?i,??????Unknown
jCHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a)??1?V?i??{V?????Unknown
vDHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a)??1?V?i????????Unknown
?EHostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1       @9       @A       @I       @a)??1?V?ix|???????Unknown
?FHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a)??1?V?i?AF????Unknown
}GHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a)??1?V?in?\????Unknown
bHHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a)??1?V?i??w?????Unknown
wIHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a)??1?V?id??
#???Unknown
yJHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1       @9       @A       @I       @a)??1?V?i?W?!.???Unknown
?KHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a)??1?V?iZBc#9???Unknown
xLHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a)??1?V?i??ڤ/D???Unknown
?MHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a)??1?V?iP?s?;O???Unknown
?NHostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a)??1?V?i?m(HZ???Unknown
?OHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a)??1?V?iF3?iTe???Unknown
?PHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a)??1?V?i??=?`p???Unknown
?QHost	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1       @9       @A       @I       @a)??1?V?i<???l{???Unknown
?RHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a)??1?V?i??o.y????Unknown
~SHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a)??1?V?i2Ip?????Unknown
?THostReluGrad",gradient_tape/sequential_7/dense_23/ReluGrad(1       @9       @A       @I       @a)??1?V?i????????Unknown
?UHostReadVariableOp",sequential_7/dense_24/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a)??1?V?i(?9??????Unknown
vVHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a)??1?F?i?6$????Unknown
aWHostIdentity"Identity(1      ??9      ??A      ??I      ??a)??1?F?i???4?????Unknown?
TXHostMul"Mul(1      ??9      ??A      ??I      ??a)??1?F?ib??U0????Unknown
uYHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ??9      ??A      ??I      ??a)??1?F?i _kv?????Unknown
uZHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a)??1?F?i??7?<????Unknown
w[HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a)??1?F?i?$??????Unknown
?\HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a)??1?F?iZ???H????Unknown
?]Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a)??1?F?i????????Unknown
?^HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a)??1?F?i?LiU????Unknown
?_HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a)??1?F?i??5;?????Unknown
?`HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a)??1?F?iR\a????Unknown
?aHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a)??1?F?iu?|?????Unknown
?bHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      ??9      ??A      ??I      ??a)??1?F?i?ך?m????Unknown
?cHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a)??1?F?i?:g??????Unknown
?dHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a)??1?F?iJ?3?y????Unknown
?eHostReluGrad",gradient_tape/sequential_7/dense_22/ReluGrad(1      ??9      ??A      ??I      ??a)??1?F?i     ???Unknown
ifHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i     ???Unknown
[gHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i     ???Unknown2CPU