"?n
BHostIDLE"IDLE1    ?J?@A    ?J?@a?Wټ?y??i?Wټ?y???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a?C??-F??iF su?B???Unknown?
dHostDataset"Iterator::Model(1      H@9      H@A      H@I      H@aSt,n?r?i/yO??f???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      F@9      F@A     ?C@I     ?C@a=s?Jm?il??s?????Unknown
iHostWriteSummary"WriteSummary(1      <@9      <@A      <@I      <@a???ոe?i?_?,?????Unknown?
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      2@9      2@A      2@I      2@a|?B??	[?iKk#?????Unknown
zHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      1@9      1@A      1@I      1@aˤiq`?Y?i???E????Unknown
tHost_FusedMatMul"sequential_7/dense_22/Relu(1      0@9      0@A      0@I      0@a??=?X?ik~B=J????Unknown
?	HostReadVariableOp"+sequential_7/dense_23/MatMul/ReadVariableOp(1      0@9      0@A      0@I      0@a??=?X?i?F??N????Unknown
?
HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      .@9      .@A      .@I      .@ag??	F?V?i?"?ɒ????Unknown
tHost_FusedMatMul"sequential_7/dense_23/Relu(1      ,@9      ,@A      ,@I      ,@a???ոU?i?Q?????Unknown
rHostSigmoid"sequential_7/dense_25/Sigmoid(1      ,@9      ,@A      ,@I      ,@a???ոU?i
???????Unknown
\HostGreater"Greater(1      *@9      *@A      *@I      *@a~?+?S?i??^????Unknown
?HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      (@9      (@A      (@I      (@aSt,n?R?i?ga????Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a|?B??	K?i?j-?#???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a|?B??	K?i[??^????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_7/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a|?B??	K?i ڨ???Unknown
`HostGatherV2"
GatherV2_1(1       @9       @A       @I       @a??=?H?i.p?????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1       @9       @A       @I       @a??=?H?iUԞC????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a??=?H?i|8nx?$???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1       @9       @A       @I       @a??=?H?i??=??*???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1       @9       @A       @I       @a??=?H?i? ??0???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a??=?H?i?d??6???Unknown
?HostMatMul",gradient_tape/sequential_7/dense_23/MatMul_1(1       @9       @A       @I       @a??=?H?iɫK?<???Unknown
~HostMatMul"*gradient_tape/sequential_7/dense_24/MatMul(1       @9       @A       @I       @a??=?H?i?-{??B???Unknown
?HostMatMul",gradient_tape/sequential_7/dense_24/MatMul_1(1       @9       @A       @I       @a??=?H?if?J??H???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a???ոE?i	???M???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a???ոE?i????@S???Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @a???ոE?iL???X???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_6/ResourceApplyGradientDescent(1      @9      @A      @I      @aSt,n?B?ii??']???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @aSt,n?B?i?"υa???Unknown
? HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @aSt,n?B?i???vf???Unknown
?!HostBiasAddGrad"7gradient_tape/sequential_7/dense_25/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aSt,n?B?i?$Y?j???Unknown
~"HostMatMul"*gradient_tape/sequential_7/dense_25/MatMul(1      @9      @A      @I      @aSt,n?B?iݯ??
o???Unknown
?#HostReadVariableOp",sequential_7/dense_25/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aSt,n?B?i?:?m?s???Unknown
?$HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a???>?i?ّ?Mw???Unknown
?%HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a???>?i*x?/{???Unknown
?&HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a???>?i????~???Unknown
~'HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a???>?iZ????????Unknown
?(HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a???>?i?S?RS????Unknown
~)HostMatMul"*gradient_tape/sequential_7/dense_22/MatMul(1      @9      @A      @I      @a???>?i????????Unknown
~*HostMatMul"*gradient_tape/sequential_7/dense_23/MatMul(1      @9      @A      @I      @a???>?i"??֍???Unknown
?+HostBiasAddGrad"7gradient_tape/sequential_7/dense_24/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???>?i?/?u?????Unknown
v,HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a??=?8?i????????Unknown
V-HostMean"Mean(1      @9      @A      @I      @a??=?8?i??l??????Unknown
?.HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a??=?8?i?E?Ě????Unknown
v/HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a??=?8?i?;ߛ????Unknown
b0HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a??=?8?i????????Unknown
?1Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a??=?8?i,\?????Unknown
?2HostBiasAddGrad"7gradient_tape/sequential_7/dense_22/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??=?8?i?s.?????Unknown
?3HostBiasAddGrad"7gradient_tape/sequential_7/dense_23/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??=?8?iR??H?????Unknown
?4HostMatMul",gradient_tape/sequential_7/dense_25/MatMul_1(1      @9      @A      @I      @a??=?8?ierBc?????Unknown
w5Host_FusedMatMul"sequential_7/dense_25/BiasAdd(1      @9      @A      @I      @a??=?8?ix$?}?????Unknown
t6HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @aSt,n?2?i?wQ?????Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @aSt,n?2?i??E%$????Unknown
X8HostEqual"Equal(1      @9      @A      @I      @aSt,n?2?i%u?d????Unknown
j9HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @aSt,n?2?i?:?̥????Unknown
v:HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @aSt,n?2?iC ???????Unknown
v;HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @aSt,n?2?i??|t'????Unknown
v<HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @aSt,n?2?ia?JHh????Unknown
~=HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @aSt,n?2?i?P?????Unknown
?>HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @aSt,n?2?i???????Unknown
??HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @aSt,n?2?iܳ?*????Unknown
?@HostReadVariableOp",sequential_7/dense_22/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aSt,n?2?i????k????Unknown
?AHostReadVariableOp",sequential_7/dense_23/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aSt,n?2?i,gOk?????Unknown
?BHostReadVariableOp"+sequential_7/dense_24/MatMul/ReadVariableOp(1      @9      @A      @I      @aSt,n?2?i?,??????Unknown
tCHost_FusedMatMul"sequential_7/dense_24/Relu(1      @9      @A      @I      @aSt,n?2?iJ??.????Unknown
vDHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a??=?(?iT???????Unknown
VEHostCast"Cast(1       @9       @A       @I       @a??=?(?i^?R-/????Unknown
XFHostCast"Cast_3(1       @9       @A       @I       @a??=?(?ih}???????Unknown
XGHostCast"Cast_5(1       @9       @A       @I       @a??=?(?irV?G0????Unknown
uHHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a??=?(?i|/?԰????Unknown
|IHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a??=?(?i?"b1????Unknown
dJHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a??=?(?i??U??????Unknown
rKHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a??=?(?i???|2????Unknown
vLHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a??=?(?i???	?????Unknown
|MHostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a??=?(?i?l??3????Unknown
?NHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a??=?(?i?E%$?????Unknown
}OHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a??=?(?i?Y?4????Unknown
`PHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a??=?(?i???>?????Unknown
wQHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a??=?(?i????5????Unknown
?RHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a??=?(?i???X?????Unknown
?SHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a??=?(?i??(?6????Unknown
?THostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a??=?(?i?[\s?????Unknown
?UHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a??=?(?i?4? 8????Unknown
?VHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a??=?(?ič?????Unknown
?WHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a??=?(?i??9????Unknown
~XHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a??=?(?i?+??????Unknown
?YHostReluGrad",gradient_tape/sequential_7/dense_23/ReluGrad(1       @9       @A       @I       @a??=?(?i&?_5:????Unknown
?ZHostReluGrad",gradient_tape/sequential_7/dense_24/ReluGrad(1       @9       @A       @I       @a??=?(?i0r?º????Unknown
?[HostReadVariableOp"+sequential_7/dense_22/MatMul/ReadVariableOp(1       @9       @A       @I       @a??=?(?i:K?O;????Unknown
v\HostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a??=??i?7a??????Unknown
X]HostCast"Cast_4(1      ??9      ??A      ??I      ??a??=??iD$?ܻ????Unknown
a^HostIdentity"Identity(1      ??9      ??A      ??I      ??a??=??i??#|????Unknown?
T_HostMul"Mul(1      ??9      ??A      ??I      ??a??=??iN?.j<????Unknown
s`HostReadVariableOp"SGD/Cast/ReadVariableOp(1      ??9      ??A      ??I      ??a??=??i??Ȱ?????Unknown
uaHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a??=??iX?b??????Unknown
wbHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??=??i???=}????Unknown
xcHostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??a??=??ib???=????Unknown
?dHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a??=??i??0??????Unknown
?eHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a??=??il???????Unknown
?fHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a??=??i?tdX~????Unknown
?gHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a??=??iva??>????Unknown
?hHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a??=??i?M???????Unknown
?iHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a??=??i?:2,?????Unknown
?jHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a??=??i'?r????Unknown
?kHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a??=??i?f??????Unknown
?lHostReluGrad",gradient_tape/sequential_7/dense_22/ReluGrad(1      ??9      ??A      ??I      ??a??=??i     ???Unknown
?mHostReadVariableOp",sequential_7/dense_24/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a??=??iI?L#` ???Unknown
?nHostReadVariableOp"+sequential_7/dense_25/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a??=??i???F? ???Unknown
LoHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i???F? ???Unknown
WpHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(i???F? ???Unknown*?m
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a?/?????i?/??????Unknown?
dHostDataset"Iterator::Model(1      H@9      H@A      H@I      H@a?n??՟?i?GJȩ???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      F@9      F@A     ?C@I     ?C@afs?P?ݙ?i??d?x???Unknown
iHostWriteSummary"WriteSummary(1      <@9      <@A      <@I      <@a???葒?i??ͥC???Unknown?
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      2@9      2@A      2@I      2@a??r*???i???O?l???Unknown
zHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      1@9      1@A      1@I      1@a?P?ݙ???i???????Unknown
tHost_FusedMatMul"sequential_7/dense_22/Relu(1      0@9      0@A      0@I      0@a??H	9??iU?/?????Unknown
?HostReadVariableOp"+sequential_7/dense_23/MatMul/ReadVariableOp(1      0@9      0@A      0@I      0@a??H	9??i??S?p???Unknown
?	HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      .@9      .@A      .@I      .@a??d?x???i?#%?T????Unknown
t
Host_FusedMatMul"sequential_7/dense_23/Relu(1      ,@9      ,@A      ,@I      ,@a???葂?i	z???
???Unknown
rHostSigmoid"sequential_7/dense_25/Sigmoid(1      ,@9      ,@A      ,@I      ,@a???葂?iA?#%?T???Unknown
\HostGreater"Greater(1      *@9      *@A      *@I      *@a?LF?W>??iu?P?ݙ???Unknown
?HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      (@9      (@A      (@I      (@a?n????i??+??????Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a??r*?w?i???H	???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a??r*?w?i??H	9???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_7/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a??r*?w?i5؝?h???Unknown
`HostGatherV2"
GatherV2_1(1       @9       @A       @I       @a??H	9u?i1j?;????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1       @9       @A       @I       @a??H	9u?iQ?­????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a??H	9u?iq???????Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1       @9       @A       @I       @a??H	9u?i???????Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1       @9       @A       @I       @a??H	9u?i????=???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a??H	9u?iѥCvg???Unknown
?HostMatMul",gradient_tape/sequential_7/dense_23/MatMul_1(1       @9       @A       @I       @a??H	9u?i????????Unknown
~HostMatMul"*gradient_tape/sequential_7/dense_24/MatMul(1       @9       @A       @I       @a??H	9u?ivg2Z????Unknown
?HostMatMul",gradient_tape/sequential_7/dense_24/MatMul_1(1       @9       @A       @I       @a??H	9u?i1^?D?????Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a?????r?iM	9????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?????r?ii?x?1???Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @a?????r?i?_??7V???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_6/ResourceApplyGradientDescent(1      @9      @A      @I      @a?n???o?i?ͥCv???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a?n???o?i?;???????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?n???o?iͩ?_?????Unknown
? HostBiasAddGrad"7gradient_tape/sequential_7/dense_25/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?n???o?i?n??????Unknown
~!HostMatMul"*gradient_tape/sequential_7/dense_25/MatMul(1      @9      @A      @I      @a?n???o?i??[{c????Unknown
?"HostReadVariableOp",sequential_7/dense_25/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?n???o?i?H	9???Unknown
?#HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a?1?K?j?i)%?T?/???Unknown
?$HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a?1?K?j?i=V?GJ???Unknown
?%HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a?1?K?j?iQ???d???Unknown
~&HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?1?K?j?ie??7V???Unknown
?'HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?1?K?j?iy?P?ݙ???Unknown
~(HostMatMul"*gradient_tape/sequential_7/dense_22/MatMul(1      @9      @A      @I      @a?1?K?j?i???d????Unknown
~)HostMatMul"*gradient_tape/sequential_7/dense_23/MatMul(1      @9      @A      @I      @a?1?K?j?i?K??????Unknown
?*HostBiasAddGrad"7gradient_tape/sequential_7/dense_24/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?1?K?j?i?|"fs????Unknown
v+HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a??H	9e?i?pko?????Unknown
V,HostMean"Mean(1      @9      @A      @I      @a??H	9e?i?d?x????Unknown
?-HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a??H	9e?i?X??)???Unknown
v.HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a??H	9e?i?LF?W>???Unknown
b/HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a??H	9e?iA???S???Unknown
?0Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a??H	9e?i5؝?h???Unknown
?1HostBiasAddGrad"7gradient_tape/sequential_7/dense_22/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??H	9e?i%)!?~???Unknown
?2HostBiasAddGrad"7gradient_tape/sequential_7/dense_23/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??H	9e?i5j?;????Unknown
?3HostMatMul",gradient_tape/sequential_7/dense_25/MatMul_1(1      @9      @A      @I      @a??H	9e?iE??t????Unknown
w4Host_FusedMatMul"sequential_7/dense_25/BiasAdd(1      @9      @A      @I      @a??H	9e?iU?­????Unknown
t5HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?n???_?ia????????Unknown
v6HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?n???_?ims?P?????Unknown
X7HostEqual"Equal(1      @9      @A      @I      @a?n???_?iy*?n????Unknown
j8HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a?n???_?i????X????Unknown
v9HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?n???_?i??ͥC???Unknown
v:HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a?n???_?i?O?l.???Unknown
v;HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?n???_?i??3-???Unknown
~<HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a?n???_?i????=???Unknown
?=HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a?n???_?i?t???L???Unknown
?>HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a?n???_?i?+???\???Unknown
??HostReadVariableOp",sequential_7/dense_22/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?n???_?i???O?l???Unknown
?@HostReadVariableOp",sequential_7/dense_23/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?n???_?i噌?|???Unknown
?AHostReadVariableOp"+sequential_7/dense_24/MatMul/ReadVariableOp(1      @9      @A      @I      @a?n???_?i?P?ݙ????Unknown
tBHost_FusedMatMul"sequential_7/dense_24/Relu(1      @9      @A      @I      @a?n???_?i?z??????Unknown
vCHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a??H	9U?i?)!????Unknown
VDHostCast"Cast(1       @9       @A       @I       @a??H	9U?i?­?????Unknown
XEHostCast"Cast_3(1       @9       @A       @I       @a??H	9U?ivg2Z????Unknown
XFHostCast"Cast_5(1       @9       @A       @I       @a??H	9U?i???????Unknown
uGHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a??H	9U?i%j?;?????Unknown
|HHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a??H	9U?i-?T?/????Unknown
dIHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a??H	9U?i5^?D?????Unknown
rJHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a??H	9U?i=؝?h????Unknown
vKHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a??H	9U?iERBN????Unknown
|LHostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a??H	9U?iM??ҡ???Unknown
?MHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a??H	9U?iUF?W>???Unknown
}NHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a??H	9U?i]?/?????Unknown
`OHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a??H	9U?ie:?`w&???Unknown
wPHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a??H	9U?im?x?1???Unknown
?QHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a??H	9U?iu.j?;???Unknown
?RHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a??H	9U?i}???LF???Unknown
?SHostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a??H	9U?i?"fs?P???Unknown
?THostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a??H	9U?i??
??[???Unknown
?UHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a??H	9U?i??|"f???Unknown
?VHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a??H	9U?i??S?p???Unknown
~WHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a??H	9U?i?
??[{???Unknown
?XHostReluGrad",gradient_tape/sequential_7/dense_23/ReluGrad(1       @9       @A       @I       @a??H	9U?i???
?????Unknown
?YHostReluGrad",gradient_tape/sequential_7/dense_24/ReluGrad(1       @9       @A       @I       @a??H	9U?i??@??????Unknown
?ZHostReadVariableOp"+sequential_7/dense_22/MatMul/ReadVariableOp(1       @9       @A       @I       @a??H	9U?i?x?1????Unknown
v[HostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a??H	9E?i??7V????Unknown
X\HostCast"Cast_4(1      ??9      ??A      ??I      ??a??H	9E?i????ͥ???Unknown
a]HostIdentity"Identity(1      ??9      ??A      ??I      ??a??H	9E?i?/??????Unknown?
T^HostMul"Mul(1      ??9      ??A      ??I      ??a??H	9E?i?l.j????Unknown
s_HostReadVariableOp"SGD/Cast/ReadVariableOp(1      ??9      ??A      ??I      ??a??H	9E?iѩ?_?????Unknown
u`HostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a??H	9E?i??ҡ????Unknown
waHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??H	9E?i?#%?T????Unknown
xbHostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??a??H	9E?i?`w&?????Unknown
?cHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a??H	9E?i???h?????Unknown
?dHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a??H	9E?i????????Unknown
?eHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a??H	9E?i?n??????Unknown
?fHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a??H	9E?i?T?/?????Unknown
?gHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a??H	9E?i??r*????Unknown
?hHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a??H	9E?i??d?x????Unknown
?iHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a??H	9E?i????????Unknown
?jHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a??H	9E?i?H	9????Unknown
?kHostReluGrad",gradient_tape/sequential_7/dense_22/ReluGrad(1      ??9      ??A      ??I      ??a??H	9E?i?[{c????Unknown
?lHostReadVariableOp",sequential_7/dense_24/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a??H	9E?ií??????Unknown
?mHostReadVariableOp"+sequential_7/dense_25/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a??H	9E?i     ???Unknown
LnHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i     ???Unknown
WoHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(i     ???Unknown2CPU