"?P
BHostIDLE"IDLE1     ??@A     ??@a#???E???i#???E????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a??ZQ??i?????V???Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      E@9      E@A     ?D@I     ?D@aCxP?? r?i???U?z???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      ;@9      ;@A      ;@I      ;@a??r*?g?i?8?ʒ???Unknown
iHostWriteSummary"WriteSummary(1      :@9      :@A      :@I      :@a?f??f?i??GJȩ???Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1      8@9      8@A      8@I      8@a??H	9e?i???S????Unknown
vHost_FusedMatMul"sequential_65/dense_202/Relu(1      3@9      3@A      3@I      3@aE7?Y'?`?i?U?z?????Unknown
XHostEqual"Equal(1      2@9      2@A      2@I      2@a?n???_?i??A?????Unknown
?	HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      &@9      &@A      &@I      &@aB?߂HtS?i?|"fs????Unknown
d
HostDataset"Iterator::Model(1      A@9      A@A      $@I      $@a?b˼??Q?ia? *K????Unknown
uHostMul"$gradient_tape/mean_squared_error/Mul(1      $@9      $@A      $@I      $@a?b˼??Q?iH??"????Unknown
?HostMatMul".gradient_tape/sequential_65/dense_203/MatMul_1(1      $@9      $@A      $@I      $@a?b˼??Q?ií??????Unknown
?HostReadVariableOp".sequential_65/dense_203/BiasAdd/ReadVariableOp(1      $@9      $@A      $@I      $@a?b˼??Q?it?u????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      "@9      "@A      "@I      "@a?n???O?i?n?????Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a?n???O?i?ʒ<????Unknown
eHost
LogicalAnd"
LogicalAnd(1      "@9      "@A      "@I      "@a?n???O?i&??$???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a?n???O?i????,???Unknown
?HostMatMul",gradient_tape/sequential_65/dense_204/MatMul(1      "@9      "@A      "@I      "@a?n???O?i?g?4???Unknown
gHostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@a?n???O?i?8?ʒ<???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a?jEaLL?i???ͥC???Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1       @9       @A       @I       @a?jEaLL?iN۰иJ???Unknown
?HostBiasAddGrad"9gradient_tape/sequential_65/dense_204/BiasAdd/BiasAddGrad(1       @9       @A       @I       @a?jEaLL?i?,???Q???Unknown
?HostCast"Smean_squared_error/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1       @9       @A       @I       @a?jEaLL?i~???X???Unknown
?HostMatMul",gradient_tape/sequential_65/dense_202/MatMul(1      @9      @A      @I      @a=?Պ?H?i3Ŗy_???Unknown
?HostMatMul",gradient_tape/sequential_65/dense_203/MatMul(1      @9      @A      @I      @a=?Պ?H?ibL@e???Unknown
vHost_FusedMatMul"sequential_65/dense_203/Relu(1      @9      @A      @I      @a=?Պ?H?i?S?pk???Unknown
yHost_FusedMatMul"sequential_65/dense_204/BiasAdd(1      @9      @A      @I      @a=?Պ?H?i???a?q???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a??H	9E?i????v???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a??H	9E?i?[?=|???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a??H	9E?i?Q?(?????Unknown
?HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1      @9      @A      @I      @a??H	9E?iЎ?jچ???Unknown
v HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?b˼??A?i???LF????Unknown
\!HostGreater"Greater(1      @9      @A      @I      @a?b˼??A?i???.?????Unknown
?"HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a?b˼??A?i['?????Unknown
?#HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a?b˼??A?i4Z???????Unknown
?$HostMatMul".gradient_tape/sequential_65/dense_204/MatMul_1(1      @9      @A      @I      @a?b˼??A?i????????Unknown
t%HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?jEaL<?i??7V????Unknown
u&HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?jEaL<?ig???????Unknown
?'HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?jEaL<?iPY?????Unknown
V(HostSum"Sum_2(1      @9      @A      @I      @a?jEaL<?i?/??????Unknown
?)HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1      @9      @A      @I      @a?jEaL<?inXh\?????Unknown
}*HostMaximum"(gradient_tape/mean_squared_error/Maximum(1      @9      @A      @I      @a?jEaL<?i???.????Unknown
?+HostBiasAddGrad"9gradient_tape/sequential_65/dense_202/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?jEaL<?iȩ?_?????Unknown
?,HostBiasAddGrad"9gradient_tape/sequential_65/dense_203/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?jEaL<?iu??A????Unknown
i-HostMean"mean_squared_error/Mean(1      @9      @A      @I      @a?jEaL<?i"??b˼???Unknown
?.HostSquaredDifference"$mean_squared_error/SquaredDifference(1      @9      @A      @I      @a?jEaL<?i?#%?T????Unknown
u/HostSum"$mean_squared_error/weighted_loss/Sum(1      @9      @A      @I      @a?jEaL<?i|L?e?????Unknown
?0HostReadVariableOp".sequential_65/dense_204/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?jEaL<?i)u=?g????Unknown
v1HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a??H	95?i??f????Unknown
V2HostCast"Cast(1      @9      @A      @I      @a??H	95?i-??)?????Unknown
X3HostCast"Cast_3(1      @9      @A      @I      @a??H	95?i?иJ]????Unknown
X4HostCast"Cast_4(1      @9      @A      @I      @a??H	95?i1??k????Unknown
b5HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a??H	95?i???????Unknown
?6HostReluGrad".gradient_tape/sequential_65/dense_203/ReluGrad(1      @9      @A      @I      @a??H	95?i5,4?R????Unknown
?7HostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1      @9      @A      @I      @a??H	95?i?J]??????Unknown
v8HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a?jEaL,?i_#??????Unknown
X9HostCast"Cast_5(1       @9       @A       @I       @a?jEaL,?ies?P?????Unknown
T:HostMul"Mul(1       @9       @A       @I       @a?jEaL,?i???H????Unknown
s;HostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a?jEaL,?i?u?????Unknown
|<HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a?jEaL,?ij?;??????Unknown
`=HostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?jEaL,?i??T?????Unknown
w>HostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?jEaL,?i??[????Unknown
w?HostCast"%gradient_tape/mean_squared_error/Cast(1       @9       @A       @I       @a?jEaL,?io???????Unknown
u@HostSum"$gradient_tape/mean_squared_error/Sum(1       @9       @A       @I       @a?jEaL,?i?T??????Unknown
AHostFloorDiv")gradient_tape/mean_squared_error/floordiv(1       @9       @A       @I       @a?jEaL,?iW?????Unknown
wBHostMul"&gradient_tape/mean_squared_error/mul_1(1       @9       @A       @I       @a?jEaL,?it*?n????Unknown
uCHostSub"$gradient_tape/mean_squared_error/sub(1       @9       @A       @I       @a?jEaL,?i?>??2????Unknown
}DHostRealDiv"(gradient_tape/mean_squared_error/truediv(1       @9       @A       @I       @a?jEaL,?i"Sl??????Unknown
?EHostSigmoidGrad"9gradient_tape/sequential_65/dense_204/Sigmoid/SigmoidGrad(1       @9       @A       @I       @a?jEaL,?iyg2Z?????Unknown
?FHostReadVariableOp"-sequential_65/dense_202/MatMul/ReadVariableOp(1       @9       @A       @I       @a?jEaL,?i?{??????Unknown
tGHostSigmoid"sequential_65/dense_204/Sigmoid(1       @9       @A       @I       @a?jEaL,?i'???E????Unknown
vHHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?jEaL?iR?!<(????Unknown
aIHostIdentity"Identity(1      ??9      ??A      ??I      ??a?jEaL?i}???
????Unknown?
?JHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a?jEaL?i?????????Unknown
VKHostMean"Mean(1      ??9      ??A      ??I      ??a?jEaL?iӸJ]?????Unknown
uLHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a?jEaL?i?­??????Unknown
wMHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?jEaL?i)??????Unknown
yNHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?jEaL?iT?s~v????Unknown
?OHostReluGrad".gradient_tape/sequential_65/dense_202/ReluGrad(1      ??9      ??A      ??I      ??a?jEaL?i???X????Unknown
|PHostDivNoNan"&mean_squared_error/weighted_loss/value(1      ??9      ??A      ??I      ??a?jEaL?i??9?;????Unknown
?QHostReadVariableOp".sequential_65/dense_202/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a?jEaL?i????????Unknown
?RHostReadVariableOp"-sequential_65/dense_203/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?jEaL?i      ???Unknown
?SHostReadVariableOp"-sequential_65/dense_204/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?jEaL?i?10q ???Unknown*?O
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a-?-???i-?-????Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      E@9      E@A     ?D@I     ?D@a????i???????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      ;@9      ;@A      ;@I      ;@az?z???i?J??J????Unknown
iHostWriteSummary"WriteSummary(1      :@9      :@A      :@I      :@al?l???i6Vc5Vc???Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1      8@9      8@A      8@I      8@aPP??i?`?`???Unknown
vHost_FusedMatMul"sequential_65/dense_202/Relu(1      3@9      3@A      3@I      3@a
?
???ii?i????Unknown
XHostEqual"Equal(1      2@9      2@A      2@I      2@a ??????i?p?p???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      &@9      &@A      &@I      &@a4A4A??i?u[?u[???Unknown
d	HostDataset"Iterator::Model(1      A@9      A@A      $@I      $@a????iz?z????Unknown
u
HostMul"$gradient_tape/mean_squared_error/Mul(1      $@9      $@A      $@I      $@a????ix~?w~????Unknown
?HostMatMul".gradient_tape/sequential_65/dense_203/MatMul_1(1      $@9      $@A      $@I      $@a????i؂-؂-???Unknown
?HostReadVariableOp".sequential_65/dense_203/BiasAdd/ReadVariableOp(1      $@9      $@A      $@I      $@a????i8?s8?s???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      "@9      "@A      "@I      "@a ?????i(??(?????Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a ?????i???????Unknown
eHost
LogicalAnd"
LogicalAnd(1      "@9      "@A      "@I      "@a ?????i?0	?0???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a ?????i??o??o???Unknown
?HostMatMul",gradient_tape/sequential_65/dense_204/MatMul(1      "@9      "@A      "@I      "@a ?????i蚮隮???Unknown
gHostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@a ?????i؞?ٞ????Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a??|?iX?%Z?%???Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1       @9       @A       @I       @a??|?iإ]ڥ]???Unknown
?HostBiasAddGrad"9gradient_tape/sequential_65/dense_204/BiasAdd/BiasAddGrad(1       @9       @A       @I       @a??|?iX??Z?????Unknown
?HostCast"Smean_squared_error/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1       @9       @A       @I       @a??|?iج?ڬ????Unknown
?HostMatMul",gradient_tape/sequential_65/dense_202/MatMul(1      @9      @A      @I      @a????x?i?????????Unknown
?HostMatMul",gradient_tape/sequential_65/dense_203/MatMul(1      @9      @A      @I      @a????x?i??/??/???Unknown
vHost_FusedMatMul"sequential_65/dense_203/Relu(1      @9      @A      @I      @a????x?i?`?`???Unknown
yHost_FusedMatMul"sequential_65/dense_204/BiasAdd(1      @9      @A      @I      @a????x?i???????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @aPPu?i?????????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @aPPu?iX??[?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @aPPu?i???????Unknown
?HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1      @9      @A      @I      @aPPu?i??9??9???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a??q?i??\??\???Unknown
\ HostGreater"Greater(1      @9      @A      @I      @a??q?i???????Unknown
?!HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a??q?i(ʢ,ʢ???Unknown
?"HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a??q?iX??\?????Unknown
?#HostMatMul".gradient_tape/sequential_65/dense_204/MatMul_1(1      @9      @A      @I      @a??q?i?????????Unknown
t$HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a??l?iH?M????Unknown
u%HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a??l?i? ? ???Unknown
?&HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a??l?i??<??<???Unknown
V'HostSum"Sum_2(1      @9      @A      @I      @a??l?i??X??X???Unknown
?(HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1      @9      @A      @I      @a??l?iH?tM?t???Unknown
})HostMaximum"(gradient_tape/mean_squared_error/Maximum(1      @9      @A      @I      @a??l?iِِ???Unknown
?*HostBiasAddGrad"9gradient_tape/sequential_65/dense_202/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??l?i?ڬ?ڬ???Unknown
?+HostBiasAddGrad"9gradient_tape/sequential_65/dense_203/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??l?i??ȍ?????Unknown
i,HostMean"mean_squared_error/Mean(1      @9      @A      @I      @a??l?iH??M?????Unknown
?-HostSquaredDifference"$mean_squared_error/SquaredDifference(1      @9      @A      @I      @a??l?i? ? ???Unknown
u.HostSum"$mean_squared_error/weighted_loss/Sum(1      @9      @A      @I      @a??l?i???????Unknown
?/HostReadVariableOp".sequential_65/dense_204/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??l?i??8??8???Unknown
v0HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @aPPe?i??M??M???Unknown
V1HostCast"Cast(1      @9      @A      @I      @aPPe?i(?b.?b???Unknown
X2HostCast"Cast_3(1      @9      @A      @I      @aPPe?ix?w~?w???Unknown
X3HostCast"Cast_4(1      @9      @A      @I      @aPPe?i?????????Unknown
b4HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @aPPe?i???????Unknown
?5HostReluGrad".gradient_tape/sequential_65/dense_203/ReluGrad(1      @9      @A      @I      @aPPe?ih??n?????Unknown
?6HostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1      @9      @A      @I      @aPPe?i??˾?????Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a??\?i??ٞ?????Unknown
X8HostCast"Cast_5(1       @9       @A       @I       @a??\?ix??~?????Unknown
T9HostMul"Mul(1       @9       @A       @I       @a??\?iX??^?????Unknown
s:HostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a??\?i8??????Unknown
|;HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a??\?i?????Unknown
`<HostDivNoNan"
div_no_nan(1       @9       @A       @I       @a??\?i???????Unknown
w=HostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a??\?i??-??-???Unknown
w>HostCast"%gradient_tape/mean_squared_error/Cast(1       @9       @A       @I       @a??\?i??;??;???Unknown
u?HostSum"$gradient_tape/mean_squared_error/Sum(1       @9       @A       @I       @a??\?i??I??I???Unknown
@HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1       @9       @A       @I       @a??\?ix?W?W???Unknown
wAHostMul"&gradient_tape/mean_squared_error/mul_1(1       @9       @A       @I       @a??\?iX?e_?e???Unknown
uBHostSub"$gradient_tape/mean_squared_error/sub(1       @9       @A       @I       @a??\?i8?s??s???Unknown
}CHostRealDiv"(gradient_tape/mean_squared_error/truediv(1       @9       @A       @I       @a??\?i???????Unknown
?DHostSigmoidGrad"9gradient_tape/sequential_65/dense_204/Sigmoid/SigmoidGrad(1       @9       @A       @I       @a??\?i?????????Unknown
?EHostReadVariableOp"-sequential_65/dense_202/MatMul/ReadVariableOp(1       @9       @A       @I       @a??\?i?????????Unknown
tFHostSigmoid"sequential_65/dense_204/Sigmoid(1       @9       @A       @I       @a??\?i?????????Unknown
vGHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a??L?i(??/?????Unknown
aHHostIdentity"Identity(1      ??9      ??A      ??I      ??a??L?i?????????Unknown?
?IHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a??L?i???????Unknown
VJHostMean"Mean(1      ??9      ??A      ??I      ??a??L?ix???????Unknown
uKHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a??L?i?????????Unknown
wLHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??L?iX??_?????Unknown
yMHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??L?i?????????Unknown
?NHostReluGrad".gradient_tape/sequential_65/dense_202/ReluGrad(1      ??9      ??A      ??I      ??a??L?i8????????Unknown
|OHostDivNoNan"&mean_squared_error/weighted_loss/value(1      ??9      ??A      ??I      ??a??L?i?????????Unknown
?PHostReadVariableOp".sequential_65/dense_202/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a??L?i???????Unknown
?QHostReadVariableOp"-sequential_65/dense_203/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a??L?i?????????Unknown
?RHostReadVariableOp"-sequential_65/dense_204/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a??L?i?????????Unknown2CPU