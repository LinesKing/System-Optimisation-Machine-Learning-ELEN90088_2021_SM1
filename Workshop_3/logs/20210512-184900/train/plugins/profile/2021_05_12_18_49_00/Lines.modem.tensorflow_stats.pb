"?c
BHostIDLE"IDLE1     ??@A     ??@a???WR???i???WR????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     P?@9     P?@A     P?@I     P?@aUUUUUU??i??I?>???Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1     ?T@9     ?T@A     ?T@I     ?T@aƆ??4}?i???fy???Unknown
iHostWriteSummary"WriteSummary(1      P@9      P@A      P@I      P@a?X?]?v?iJ?q????Unknown?
dHostDataset"Iterator::Model(1     ?[@9     ?[@A      <@I      <@apm??c?i??#[&????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      :@9      :@A      :@I      :@a??1\Lb?ik?U?r????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      :@9      :@A      6@I      6@a??`?^?iC??g?????Unknown
VHostSum"Sum_2(1      6@9      6@A      6@I      6@a??`?^?i?Gj????Unknown
l	HostIteratorGetNext"IteratorGetNext(1      5@9      5@A      5@I      5@a(??]?i/?ҝ1????Unknown
s
Host_FusedMatMul"sequential_1/dense_3/Relu(1      4@9      4@A      4@I      4@a?.9?&\?i2o?D???Unknown
?HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      1@9      1@A      1@I      1@aΣ??W?i?A?;???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      ,@9      ,@A      ,@I      ,@apm??S?i;PH+???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@a?.9?&L?i㛖?%???Unknown
?HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      $@9      $@A      $@I      $@a?.9?&L?i????),???Unknown
gHostStridedSlice"strided_slice(1      $@9      $@A      $@I      $@a?.9?&L?i333333???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      "@9      "@A      "@I      "@a?C?	VI?iD???9???Unknown
eHost
LogicalAnd"
LogicalAnd(1      "@9      "@A      "@I      "@a?C?	VI?i?T?7?????Unknown?
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      "@9      "@A      "@I      "@a?C?	VI?i?eS?3F???Unknown
HostMatMul"+gradient_tape/sequential_1/dense_4/MatMul_1(1      "@9      "@A      "@I      "@a?C?	VI?i?v?<?L???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a?X?]?F?i?L%?*R???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a?X?]?F?i#???W???Unknown
}HostMatMul")gradient_tape/sequential_1/dense_3/MatMul(1       @9       @A       @I       @a?X?]?F?i#?Cm]???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @apm??C?i??oZb???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @apm??C?i?/?Gg???Unknown
}HostMatMul")gradient_tape/sequential_1/dense_5/MatMul(1      @9      @A      @I      @apm??C?i7˓?4l???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a`?U?@?i?+)?mp???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a`?U?@?ig??˦t???Unknown
}HostMatMul")gradient_tape/sequential_1/dense_4/MatMul(1      @9      @A      @I      @a`?U?@?i??S??x???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?.9?&<?i???d|???Unknown
vHostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?.9?&<?i?8?z????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?.9?&<?i{^IQn????Unknown
? HostBiasAddGrad"6gradient_tape/sequential_1/dense_3/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?.9?&<?iO??'?????Unknown
?!HostBiasAddGrad"6gradient_tape/sequential_1/dense_4/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?.9?&<?i#???w????Unknown
?"HostBiasAddGrad"6gradient_tape/sequential_1/dense_5/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?.9?&<?i??>??????Unknown
s#Host_FusedMatMul"sequential_1/dense_4/Relu(1      @9      @A      @I      @a?.9?&<?i??嫁????Unknown
v$Host_FusedMatMul"sequential_1/dense_5/BiasAdd(1      @9      @A      @I      @a?.9?&<?i???????Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?X?]?6?i?F.ח???Unknown
?&HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a?X?]?6?i???٧????Unknown
V'HostMean"Mean(1      @9      @A      @I      @a?X?]?6?i?ܷ?x????Unknown
?(HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a?X?]?6?i??p1I????Unknown
?)HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a?X?]?6?i??)?????Unknown
z*HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a?X?]?6?i?????????Unknown
~+HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?X?]?6?i??4?????Unknown
v,HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?X?]?6?itT??????Unknown
b-HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a?X?]?6?i/_?\????Unknown
~.HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a?X?]?6?i?J?7-????Unknown
?/HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?X?]?6?iO5??????Unknown
?0HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?X?]?6?i_ 8?ζ???Unknown
?1HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?X?]?6?io?:?????Unknown
?2HostReadVariableOp"+sequential_1/dense_5/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?X?]?6?i???o????Unknown
t3HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a`?U?0?i˦tg?????Unknown
v4HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a`?U?0?iW???????Unknown
V5HostCast"Cast(1      @9      @A      @I      @a`?U?0?ic
i?????Unknown
\6HostGreater"Greater(1      @9      @A      @I      @a`?U?0?i?????????Unknown
s7HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a`?U?0?i?g?j?????Unknown
u8HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a`?U?0?iGj?????Unknown
|9HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      @9      @A      @I      @a`?U?0?i??4l7????Unknown
v:HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a`?U?0?i?x??S????Unknown
v;HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a`?U?0?i+)?mp????Unknown
?<Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a`?U?0?iwٔ??????Unknown
?=Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a`?U?0?iÉ_o?????Unknown
?>HostReadVariableOp"+sequential_1/dense_3/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a`?U?0?i:*??????Unknown
??HostReadVariableOp"*sequential_1/dense_3/MatMul/ReadVariableOp(1      @9      @A      @I      @a`?U?0?i[??p?????Unknown
?@HostReadVariableOp"*sequential_1/dense_5/MatMul/ReadVariableOp(1      @9      @A      @I      @a`?U?0?i?????????Unknown
vAHostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a?X?]?&?i/?Gg????Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a?X?]?&?i??x??????Unknown
XCHostCast"Cast_3(1       @9       @A       @I       @a?X?]?&?i??T?7????Unknown
XDHostEqual"Equal(1       @9       @A       @I       @a?X?]?&?i?p1I?????Unknown
TEHostMul"Mul(1       @9       @A       @I       @a?X?]?&?iO??????Unknown
jFHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a?X?]?&?i?[??p????Unknown
rGHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a?X?]?&?i_??J?????Unknown
vHHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?X?]?&?i?F??A????Unknown
}IHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a?X?]?&?io???????Unknown
`JHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?X?]?&?i?1\L????Unknown
wKHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?X?]?&?i?8?z????Unknown
yLHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1       @9       @A       @I       @a?X?]?&?i??????Unknown
xMHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?X?]?&?i???MK????Unknown
?NHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a?X?]?&?iΣ?????Unknown
?OHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a?X?]?&?i?}??????Unknown
?PHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?X?]?&?i'??O?????Unknown
?QHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a?X?]?&?i?hc??????Unknown
~RHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?X?]?&?i7???T????Unknown
?SHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?X?]?&?i?SQ?????Unknown
THostMatMul"+gradient_tape/sequential_1/dense_5/MatMul_1(1       @9       @A       @I       @a?X?]?&?iG???%????Unknown
?UHostReadVariableOp"+sequential_1/dense_4/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?X?]?&?i?>???????Unknown
?VHostReadVariableOp"*sequential_1/dense_4/MatMul/ReadVariableOp(1       @9       @A       @I       @a?X?]?&?iW??R?????Unknown
qWHostSigmoid"sequential_1/dense_5/Sigmoid(1       @9       @A       @I       @a?X?]?&?i?)??^????Unknown
XXHostCast"Cast_4(1      ??9      ??A      ??I      ??a?X?]??i?d|?????Unknown
XYHostCast"Cast_5(1      ??9      ??A      ??I      ??a?X?]??ig?j??????Unknown
aZHostIdentity"Identity(1      ??9      ??A      ??I      ??a?X?]??i+?X){????Unknown?
d[HostAddN"SGD/gradients/AddN(1      ??9      ??A      ??I      ??a?X?]??i?GT/????Unknown
?\HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??a?X?]??i?O5?????Unknown
?]Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a?X?]??iw?#??????Unknown
?^HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?X?]??i;??K????Unknown
?_HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      ??9      ??A      ??I      ??a?X?]??i?????????Unknown
?`HostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a?X?]??ibwZ ???Unknown
?aHostReluGrad"+gradient_tape/sequential_1/dense_3/ReluGrad(1      ??9      ??A      ??I      ??a?X?]??i?:?*? ???Unknown
?bHostReluGrad"+gradient_tape/sequential_1/dense_4/ReluGrad(1      ??9      ??A      ??I      ??a?X?]??i&Xe@???Unknown
HcHostReadVariableOp"div_no_nan/ReadVariableOp(i&Xe@???Unknown
JdHostReadVariableOp"div_no_nan/ReadVariableOp_1(i&Xe@???Unknown
WeHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(i&Xe@???Unknown
[fHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i&Xe@???Unknown
YgHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i&Xe@???Unknown
[hHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i&Xe@???Unknown*?c
uHostFlushSummaryWriter"FlushSummaryWriter(1     P?@9     P?@A     P?@I     P?@ai'8}D??ii'8}D???Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1     ?T@9     ?T@A     ?T@I     ?T@a0?4??`??i?p(?????Unknown
iHostWriteSummary"WriteSummary(1      P@9      P@A      P@I      P@aN?W??i?eu?/???Unknown?
dHostDataset"Iterator::Model(1     ?[@9     ?[@A      <@I      <@aEd?62̑?i?@?_????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      :@9      :@A      :@I      :@aeo??????iy?$?B???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      :@9      :@A      6@I      6@aH??????i??? t????Unknown
VHostSum"Sum_2(1      6@9      6@A      6@I      6@aH??????i?? ?R"???Unknown
lHostIteratorGetNext"IteratorGetNext(1      5@9      5@A      5@I      5@ahyRK???i-?j ????Unknown
s	Host_FusedMatMul"sequential_1/dense_3/Relu(1      4@9      4@A      4@I      4@a?!g?l??i?e???????Unknown
?
HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      1@9      1@A      1@I      1@a?B1????i?*??AI???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      ,@9      ,@A      ,@I      ,@aEd?62́?iP?ir????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@a?!g?ly?i???L????Unknown
?HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      $@9      $@A      $@I      $@a?!g?ly?iִ??%????Unknown
gHostStridedSlice"strided_slice(1      $@9      $@A      $@I      $@a?!g?ly?i????(???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      "@9      "@A      "@I      "@a?7Ck??v?i?	?t?V???Unknown
eHost
LogicalAnd"
LogicalAnd(1      "@9      "@A      "@I      "@a?7Ck??v?i???c?????Unknown?
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      "@9      "@A      "@I      "@a?7Ck??v?iiyRK????Unknown
HostMatMul"+gradient_tape/sequential_1/dense_4/MatMul_1(1      "@9      "@A      "@I      "@a?7Ck??v?iٜOA????Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @aN?Wt?iu??j????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @aN?Wt?i??k1???Unknown
}HostMatMul")gradient_tape/sequential_1/dense_3/MatMul(1       @9       @A       @I       @aN?Wt?i?X6?Z???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @aEd?62?q?ivO?"?}???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @aEd?62?q?i?F?J????Unknown
}HostMatMul")gradient_tape/sequential_1/dense_5/MatMul(1      @9      @A      @I      @aEd?62?q?i=???????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a	??9??n?i?빊e????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a	??9??n?i???)????Unknown
}HostMatMul")gradient_tape/sequential_1/dense_4/MatMul(1      @9      @A      @I      @a	??9??n?i?I-?j ???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?!g?li?i	?2??9???Unknown
vHostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?!g?li?i+8}DS???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?!g?li?iM=W?l???Unknown
?HostBiasAddGrad"6gradient_tape/sequential_1/dense_3/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?!g?li?io?B1????Unknown
? HostBiasAddGrad"6gradient_tape/sequential_1/dense_4/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?!g?li?i?MH?????Unknown
?!HostBiasAddGrad"6gradient_tape/sequential_1/dense_5/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?!g?li?i??M??????Unknown
s"Host_FusedMatMul"sequential_1/dense_4/Relu(1      @9      @A      @I      @a?!g?li?i?S?d????Unknown
v#Host_FusedMatMul"sequential_1/dense_5/BiasAdd(1      @9      @A      @I      @a?!g?li?i??X??????Unknown
v$HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @aN?Wd?iE?)?( ???Unknown
?%HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @aN?Wd?i???????Unknown
V&HostMean"Mean(1      @9      @A      @I      @aN?Wd?i?????(???Unknown
?'HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @aN?Wd?i/ ??-=???Unknown
?(HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @aN?Wd?i}n?Q???Unknown
z)HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @aN?Wd?i?>??e???Unknown
~*HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @aN?Wd?i^+3z???Unknown
v+HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @aN?Wd?ig}???????Unknown
b,HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @aN?Wd?i???T?????Unknown
~-HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @aN?Wd?i??i8????Unknown
?.HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @aN?Wd?iQ?T~?????Unknown
?/HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @aN?Wd?i??%??????Unknown
?0HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @aN?Wd?i???=????Unknown
?1HostReadVariableOp"+sequential_1/dense_5/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aN?Wd?i;9ȼ????Unknown
t2HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a	??9??^?i?e????Unknown
v3HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a	??9??^?i1?\'???Unknown
V4HostCast"Cast(1      @9      @A      @I      @a	??9??^?i????X6???Unknown
\5HostGreater"Greater(1      @9      @A      @I      @a	??9??^?i'?;??E???Unknown
s6HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a	??9??^?i?n?J?T???Unknown
u7HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a	??9??^?iFu?d???Unknown
|8HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      @9      @A      @I      @a	??9??^?i??]s???Unknown
v9HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a	??9??^?i??9?????Unknown
v:HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a	??9??^?i??K??????Unknown
?;Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a	??9??^?i	???!????Unknown
?<Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a	??9??^?i?{?(c????Unknown
?=HostReadVariableOp"+sequential_1/dense_3/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a	??9??^?i?R"x?????Unknown
?>HostReadVariableOp"*sequential_1/dense_3/MatMul/ReadVariableOp(1      @9      @A      @I      @a	??9??^?iz*???????Unknown
??HostReadVariableOp"*sequential_1/dense_5/MatMul/ReadVariableOp(1      @9      @A      @I      @a	??9??^?i?\'????Unknown
v@HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @aN?WT?i??ġR????Unknown
vAHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @aN?WT?iC!-,~????Unknown
XBHostCast"Cast_3(1       @9       @A       @I       @aN?WT?i강??????Unknown
XCHostEqual"Equal(1       @9       @A       @I       @aN?WT?i?@?@????Unknown
TDHostMul"Mul(1       @9       @A       @I       @aN?WT?i8?f? ???Unknown
jEHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @aN?WT?i?_?U,???Unknown
rFHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @aN?WT?i??7?W%???Unknown
vGHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @aN?WT?i-?j?/???Unknown
}HHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @aN?WT?i?	??9???Unknown
`IHostDivNoNan"
div_no_nan(1       @9       @A       @I       @aN?WT?i{?q?C???Unknown
wJHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @aN?WT?i".?	N???Unknown
yKHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1       @9       @A       @I       @aN?WT?iɽB?1X???Unknown
xLHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @aN?WT?ipM?]b???Unknown
?MHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @aN?WT?i???l???Unknown
?NHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @aN?WT?i?l|3?v???Unknown
?OHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @aN?WT?ie???߀???Unknown
?PHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @aN?WT?i?MH????Unknown
~QHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @aN?WT?i???6????Unknown
?RHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @aN?WT?iZ?]b????Unknown
SHostMatMul"+gradient_tape/sequential_1/dense_5/MatMul_1(1       @9       @A       @I       @aN?WT?i;?獩???Unknown
?THostReadVariableOp"+sequential_1/dense_4/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aN?WT?i???q?????Unknown
?UHostReadVariableOp"*sequential_1/dense_4/MatMul/ReadVariableOp(1       @9       @A       @I       @aN?WT?iOZX??????Unknown
qVHostSigmoid"sequential_1/dense_5/Sigmoid(1       @9       @A       @I       @aN?WT?i????????Unknown
XWHostCast"Cast_4(1      ??9      ??A      ??I      ??aN?WD?i?1?K&????Unknown
XXHostCast"Cast_5(1      ??9      ??A      ??I      ??aN?WD?i?y)<????Unknown
aYHostIdentity"Identity(1      ??9      ??A      ??I      ??aN?WD?ir?]?Q????Unknown?
dZHostAddN"SGD/gradients/AddN(1      ??9      ??A      ??I      ??aN?WD?iF	??g????Unknown
?[HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??aN?WD?iQ?`}????Unknown
?\Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??aN?WD?i???%?????Unknown
?]HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??aN?WD?i??.??????Unknown
?^HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      ??9      ??A      ??I      ??aN?WD?i?(c??????Unknown
?_HostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??aN?WD?ijp?u?????Unknown
?`HostReluGrad"+gradient_tape/sequential_1/dense_3/ReluGrad(1      ??9      ??A      ??I      ??aN?WD?i>??:?????Unknown
?aHostReluGrad"+gradient_tape/sequential_1/dense_4/ReluGrad(1      ??9      ??A      ??I      ??aN?WD?i	     ???Unknown
HbHostReadVariableOp"div_no_nan/ReadVariableOp(i	     ???Unknown
JcHostReadVariableOp"div_no_nan/ReadVariableOp_1(i	     ???Unknown
WdHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(i	     ???Unknown
[eHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i	     ???Unknown
YfHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i	     ???Unknown
[gHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i	     ???Unknown2CPU