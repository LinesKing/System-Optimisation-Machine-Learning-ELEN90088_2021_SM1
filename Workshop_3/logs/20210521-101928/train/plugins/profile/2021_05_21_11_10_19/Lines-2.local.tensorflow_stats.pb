"?O
BHostIDLE"IDLE1     ??@A     ??@a??Bx???i??Bx????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?@9     ?@A     ?@I     ?@a???"s??iʈ???O???Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1     ?B@9     ?B@A     ?B@I     ?B@a???T?:q?i???0Rr???Unknown
iHostWriteSummary"WriteSummary(1      B@9      B@A      B@I      B@a??\n??p?iq:?Yٓ???Unknown?
vHost_FusedMatMul"sequential_55/dense_171/Relu(1      7@9      7@A      7@I      7@a?&?p?ke?i???E????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      :@9      :@A      6@I      6@a?U?C}d?i?"?L½???Unknown
dHostDataset"Iterator::Model(1      L@9      L@A      3@I      3@aP??; ?a?i???Lt????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      2@9      2@A      2@I      2@a??\n??`?i?D?7????Unknown
?	HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      0@9      0@A      0@I      0@aə??y?]?iY??????Unknown
V
HostSum"Sum_2(1      0@9      0@A      0@I      0@aə??y?]?i&??Z????Unknown
?HostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1      *@9      *@A      *@I      *@a?LM??6X?i?X? 
???Unknown
}HostMaximum"(gradient_tape/mean_squared_error/Maximum(1      (@9      (@A      (@I      (@aW3?=ZV?if???M???Unknown
`HostDivNoNan"
div_no_nan(1      &@9      &@A      &@I      &@a?U?C}T?i??ȃ????Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a??\n??P?if??M?'???Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @aə??y?M?i̴i?a/???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1       @9       @A       @I       @aə??y?M?i2??
?6???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @aə??y?M?i??=iH>???Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1       @9       @A       @I       @aə??y?M?i???ǻE???Unknown
?HostMatMul",gradient_tape/sequential_55/dense_172/MatMul(1       @9       @A       @I       @aə??y?M?idv&/M???Unknown
?HostMatMul".gradient_tape/sequential_55/dense_172/MatMul_1(1       @9       @A       @I       @aə??y?M?i?f{??T???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a?f?r?J?i$w'[???Unknown
?HostMatMul",gradient_tape/sequential_55/dense_171/MatMul(1      @9      @A      @I      @a?f?r?J?i~˴i?a???Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @a?f?r?J?i?}Q\1h???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @aW3?=ZF?i%? ??m???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @aW3?=ZF?irf?i^s???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @aW3?=ZF?i?ڿ??x???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @aW3?=ZF?iO?w?~???Unknown
?HostMatMul",gradient_tape/sequential_55/dense_173/MatMul(1      @9      @A      @I      @aW3?=ZF?iY?^?!????Unknown
?HostCast"Smean_squared_error/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @aW3?=ZF?i?7.??????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a ?l?B?i?m0?`????Unknown
?HostBiasAddGrad"9gradient_tape/sequential_55/dense_173/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a ?l?B?i&?2?????Unknown
? HostSquaredDifference"$mean_squared_error/SquaredDifference(1      @9      @A      @I      @a ?l?B?if?4ְ????Unknown
u!HostSum"$mean_squared_error/weighted_loss/Sum(1      @9      @A      @I      @a ?l?B?i?7?X????Unknown
y"Host_FusedMatMul"sequential_55/dense_173/BiasAdd(1      @9      @A      @I      @a ?l?B?i?F9????Unknown
v#HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @aə??y?=?i?n??????Unknown
v$HostAssignAddVariableOp"AssignAddVariableOp_3(1      @9      @A      @I      @aə??y?=?iL7?jt????Unknown
?%HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @aə??y?=?i/?.????Unknown
?&HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1      @9      @A      @I      @aə??y?=?i?'??????Unknown
?'HostBiasAddGrad"9gradient_tape/sequential_55/dense_172/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aə??y?=?i?Bx?????Unknown
v(Host_FusedMatMul"sequential_55/dense_172/Relu(1      @9      @A      @I      @aə??y?=?iw'[????Unknown
?)HostReadVariableOp"-sequential_55/dense_173/MatMul/ReadVariableOp(1      @9      @A      @I      @aə??y?=?iK??????Unknown
v*HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @aW3?=Z6?iq??????Unknown
\+HostGreater"Greater(1      @9      @A      @I      @aW3?=Z6?i??{]?????Unknown
s,HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @aW3?=Z6?i?>??v????Unknown
?-HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1      @9      @A      @I      @aW3?=Z6?i??J?A????Unknown
?.HostBiasAddGrad"9gradient_tape/sequential_55/dense_171/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aW3?=Z6?i	??'????Unknown
?/HostMatMul".gradient_tape/sequential_55/dense_173/MatMul_1(1      @9      @A      @I      @aW3?=Z6?i/mk?????Unknown
|0HostDivNoNan"&mean_squared_error/weighted_loss/value(1      @9      @A      @I      @aW3?=Z6?iU'???????Unknown
?1HostReadVariableOp"-sequential_55/dense_172/MatMul/ReadVariableOp(1      @9      @A      @I      @aW3?=Z6?i{???n????Unknown
t2HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @aə??y?-?i?]??K????Unknown
V3HostCast"Cast(1       @9       @A       @I       @aə??y?-?i???(????Unknown
X4HostEqual"Equal(1       @9       @A       @I       @aə??y?-?i?U?x????Unknown
V5HostMean"Mean(1       @9       @A       @I       @aə??y?-?i??SP?????Unknown
T6HostMul"Mul(1       @9       @A       @I       @aə??y?-?i?M?'?????Unknown
u7HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @aə??y?-?iʈ??????Unknown
b8HostDivNoNan"div_no_nan_1(1       @9       @A       @I       @aə??y?-?i1F#?x????Unknown
w9HostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @aə??y?-?iK½?U????Unknown
?:HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1       @9       @A       @I       @aə??y?-?ie>X?2????Unknown
u;HostMul"$gradient_tape/mean_squared_error/Mul(1       @9       @A       @I       @aə??y?-?i??]????Unknown
<HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1       @9       @A       @I       @aə??y?-?i?6?5?????Unknown
u=HostSub"$gradient_tape/mean_squared_error/sub(1       @9       @A       @I       @aə??y?-?i??'?????Unknown
}>HostRealDiv"(gradient_tape/mean_squared_error/truediv(1       @9       @A       @I       @aə??y?-?i?.???????Unknown
??HostReluGrad".gradient_tape/sequential_55/dense_171/ReluGrad(1       @9       @A       @I       @aə??y?-?i??\??????Unknown
i@HostMean"mean_squared_error/Mean(1       @9       @A       @I       @aə??y?-?i'??_????Unknown
?AHostReadVariableOp"-sequential_55/dense_171/MatMul/ReadVariableOp(1       @9       @A       @I       @aə??y?-?i??k<????Unknown
?BHostReadVariableOp".sequential_55/dense_172/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aə??y?-?i5,C????Unknown
?CHostReadVariableOp".sequential_55/dense_173/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aə??y?-?iO???????Unknown
tDHostSigmoid"sequential_55/dense_173/Sigmoid(1       @9       @A       @I       @aə??y?-?iia??????Unknown
vEHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??aə??y??ivU.^?????Unknown
XFHostCast"Cast_3(1      ??9      ??A      ??I      ??aə??y??i???ɯ????Unknown
XGHostCast"Cast_4(1      ??9      ??A      ??I      ??aə??y??i???5?????Unknown
XHHostCast"Cast_5(1      ??9      ??A      ??I      ??aə??y??i????????Unknown
|IHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??aə??y??i?Mc{????Unknown
yJHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??aə??y??i??0yi????Unknown
wKHostCast"%gradient_tape/mean_squared_error/Cast(1      ??9      ??A      ??I      ??aə??y??i????W????Unknown
uLHostSum"$gradient_tape/mean_squared_error/Sum(1      ??9      ??A      ??I      ??aə??y??i??PF????Unknown
wMHostMul"&gradient_tape/mean_squared_error/mul_1(1      ??9      ??A      ??I      ??aə??y??i?E??4????Unknown
?NHostReluGrad".gradient_tape/sequential_55/dense_172/ReluGrad(1      ??9      ??A      ??I      ??aə??y??i??e(#????Unknown
?OHostSigmoidGrad"9gradient_tape/sequential_55/dense_173/Sigmoid/SigmoidGrad(1      ??9      ??A      ??I      ??aə??y??i??2?????Unknown
?PHostReadVariableOp".sequential_55/dense_171/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??aə??y??i     ???Unknown
4QHostIdentity"Identity(i     ???Unknown?
HRHostReadVariableOp"div_no_nan/ReadVariableOp(i     ???Unknown
JSHostReadVariableOp"div_no_nan/ReadVariableOp_1(i     ???Unknown*?N
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?@9     ?@A     ?@I     ?@a???i???i???i????Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1     ?B@9     ?B@A     ?B@I     ?B@a?Z????ip?e?p????Unknown
iHostWriteSummary"WriteSummary(1      B@9      B@A      B@I      B@ah?ű??i?#)??????Unknown?
vHost_FusedMatMul"sequential_55/dense_171/Relu(1      7@9      7@A      7@I      7@a???=>???i???1???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      :@9      :@A      6@I      6@avX??????i?3??????Unknown
dHostDataset"Iterator::Model(1      L@9      L@A      3@I      3@a	XSI(3??i??g??I???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      2@9      2@A      2@I      2@ah?ű??i?n??P????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      0@9      0@A      0@I      0@a7?꣯H??i]Y?s1???Unknown
V	HostSum"Sum_2(1      0@9      0@A      0@I      0@a7?꣯H??i??|?????Unknown
?
HostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1      *@9      *@A      *@I      *@a]?.?+??i?~??B????Unknown
}HostMaximum"(gradient_tape/mean_squared_error/Maximum(1      (@9      (@A      (@I      (@ai???v??i?>??I???Unknown
`HostDivNoNan"
div_no_nan(1      &@9      &@A      &@I      &@avX??????iC??$????Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@ah?ű~?iM??4?????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a7?꣯H{?i??$????Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1       @9       @A       @I       @a7?꣯H{?i	l??>???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @a7?꣯H{?igT?R<u???Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1       @9       @A       @I       @a7?꣯H{?i?)??ͫ???Unknown
?HostMatMul",gradient_tape/sequential_55/dense_172/MatMul(1       @9       @A       @I       @a7?꣯H{?i#?C_????Unknown
?HostMatMul".gradient_tape/sequential_55/dense_172/MatMul_1(1       @9       @A       @I       @a7?꣯H{?i?ԋp????Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @aPYm???w?i4?꣯H???Unknown
?HostMatMul",gradient_tape/sequential_55/dense_171/MatMul(1      @9      @A      @I      @aPYm???w?i??I?nx???Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @aPYm???w?i?d?
.????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @ai???vt?i?D????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @ai???vt?i?$?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @ai???vt?i?
!?"???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @ai???vt?i??(?K???Unknown
?HostMatMul",gradient_tape/sequential_55/dense_173/MatMul(1      @9      @A      @I      @ai???vt?i???/?t???Unknown
?HostCast"Smean_squared_error/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @ai???vt?iĤk7?????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a??r?mq?i??׿???Unknown
?HostBiasAddGrad"9gradient_tape/sequential_55/dense_173/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??r?mq?izo???????Unknown
?HostSquaredDifference"$mean_squared_error/SquaredDifference(1      @9      @A      @I      @a??r?mq?i?T????Unknown
u HostSum"$mean_squared_error/weighted_loss/Sum(1      @9      @A      @I      @a??r?mq?i0:??'&???Unknown
y!Host_FusedMatMul"sequential_55/dense_173/BiasAdd(1      @9      @A      @I      @a??r?mq?i?,?BH???Unknown
v"HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a7?꣯Hk?i:
?0?c???Unknown
v#HostAssignAddVariableOp"AssignAddVariableOp_3(1      @9      @A      @I      @a7?꣯Hk?i??s??~???Unknown
?$HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a7?꣯Hk?i???????Unknown
?%HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1      @9      @A      @I      @a7?꣯Hk?iGʻ?e????Unknown
?&HostBiasAddGrad"9gradient_tape/sequential_55/dense_172/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a7?꣯Hk?i??_??????Unknown
v'Host_FusedMatMul"sequential_55/dense_172/Relu(1      @9      @A      @I      @a7?꣯Hk?i????????Unknown
?(HostReadVariableOp"-sequential_55/dense_173/MatMul/ReadVariableOp(1      @9      @A      @I      @a7?꣯Hk?iT??N????Unknown
v)HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @ai???vd?iWzbҵ???Unknown
\*HostGreater"Greater(1      @9      @A      @I      @ai???vd?iZjV,0???Unknown
s+HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @ai???vd?i]Z?٢D???Unknown
?,HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1      @9      @A      @I      @ai???vd?i`J?]Y???Unknown
?-HostBiasAddGrad"9gradient_tape/sequential_55/dense_171/BiasAdd/BiasAddGrad(1      @9      @A      @I      @ai???vd?ic:N??m???Unknown
?.HostMatMul".gradient_tape/sequential_55/dense_173/MatMul_1(1      @9      @A      @I      @ai???vd?if*	e????Unknown
|/HostDivNoNan"&mean_squared_error/weighted_loss/value(1      @9      @A      @I      @ai???vd?ii??|????Unknown
?0HostReadVariableOp"-sequential_55/dense_172/MatMul/ReadVariableOp(1      @9      @A      @I      @ai???vd?il
l?????Unknown
t1HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a7?꣯H[?i??Pė????Unknown
V2HostCast"Cast(1       @9       @A       @I       @a7?꣯H[?i?"<????Unknown
X3HostEqual"Equal(1       @9       @A       @I       @a7?꣯H[?it??s?????Unknown
V4HostMean"Mean(1       @9       @A       @I       @a7?꣯H[?i???˄????Unknown
T5HostMul"Mul(1       @9       @A       @I       @a7?꣯H[?i$՘#)????Unknown
u6HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a7?꣯H[?i|?j{?????Unknown
b7HostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a7?꣯H[?iԿ<?q
???Unknown
w8HostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a7?꣯H[?i,?+???Unknown
?9HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1       @9       @A       @I       @a7?꣯H[?i?????%???Unknown
u:HostMul"$gradient_tape/mean_squared_error/Mul(1       @9       @A       @I       @a7?꣯H[?iܟ??^3???Unknown
;HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1       @9       @A       @I       @a7?꣯H[?i4??2A???Unknown
u<HostSub"$gradient_tape/mean_squared_error/sub(1       @9       @A       @I       @a7?꣯H[?i??V??N???Unknown
}=HostRealDiv"(gradient_tape/mean_squared_error/truediv(1       @9       @A       @I       @a7?꣯H[?i?(?K\???Unknown
?>HostReluGrad".gradient_tape/sequential_55/dense_171/ReluGrad(1       @9       @A       @I       @a7?꣯H[?i<u?9?i???Unknown
i?HostMean"mean_squared_error/Mean(1       @9       @A       @I       @a7?꣯H[?i?j̑?w???Unknown
?@HostReadVariableOp"-sequential_55/dense_171/MatMul/ReadVariableOp(1       @9       @A       @I       @a7?꣯H[?i?_??8????Unknown
?AHostReadVariableOp".sequential_55/dense_172/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a7?꣯H[?iDUpAݒ???Unknown
?BHostReadVariableOp".sequential_55/dense_173/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a7?꣯H[?i?JB??????Unknown
tCHostSigmoid"sequential_55/dense_173/Sigmoid(1       @9       @A       @I       @a7?꣯H[?i???%????Unknown
vDHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a7?꣯HK?i?:??????Unknown
XEHostCast"Cast_3(1      ??9      ??A      ??I      ??a7?꣯HK?iL5?Hʻ???Unknown
XFHostCast"Cast_4(1      ??9      ??A      ??I      ??a7?꣯HK?i?/?t?????Unknown
XGHostCast"Cast_5(1      ??9      ??A      ??I      ??a7?꣯HK?i?*??n????Unknown
|HHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a7?꣯HK?iP%??@????Unknown
yIHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a7?꣯HK?i???????Unknown
wJHostCast"%gradient_tape/mean_squared_error/Cast(1      ??9      ??A      ??I      ??a7?꣯HK?i?s$?????Unknown
uKHostSum"$gradient_tape/mean_squared_error/Sum(1      ??9      ??A      ??I      ??a7?꣯HK?iT\P?????Unknown
wLHostMul"&gradient_tape/mean_squared_error/mul_1(1      ??9      ??A      ??I      ??a7?꣯HK?i E|?????Unknown
?MHostReluGrad".gradient_tape/sequential_55/dense_172/ReluGrad(1      ??9      ??A      ??I      ??a7?꣯HK?i?
.?[????Unknown
?NHostSigmoidGrad"9gradient_tape/sequential_55/dense_173/Sigmoid/SigmoidGrad(1      ??9      ??A      ??I      ??a7?꣯HK?iX?-????Unknown
?OHostReadVariableOp".sequential_55/dense_171/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a7?꣯HK?i     ???Unknown
4PHostIdentity"Identity(i     ???Unknown?
HQHostReadVariableOp"div_no_nan/ReadVariableOp(i     ???Unknown
JRHostReadVariableOp"div_no_nan/ReadVariableOp_1(i     ???Unknown2CPU